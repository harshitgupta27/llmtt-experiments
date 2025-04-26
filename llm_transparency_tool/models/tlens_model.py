# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import List, Optional

import torch
import transformer_lens
import transformers
from fancy_einsum import einsum
from jaxtyping import Float, Int
from typeguard import typechecked
import streamlit as st

from llm_transparency_tool.models.transparent_llm import ModelInfo, TransparentLlm
from transformer_lens.loading_from_pretrained import MODEL_ALIASES, get_official_model_name
from llm_transparency_tool.models.layerSkipCustom import LayerSkipLlamaTransparentLlm

@st.cache_resource
def load_custom_model(model_name):
    return LayerSkipLlamaTransparentLlm(model_name) # facebook/layerskip-llama3.2-1B

@dataclass
class _RunInfo:
    tokens: Int[torch.Tensor, "batch pos"]
    logits: Float[torch.Tensor, "batch pos d_vocab"]
    cache: transformer_lens.ActivationCache


@st.cache_resource(
    max_entries=1,
    show_spinner=True,
    hash_funcs={
        transformers.PreTrainedModel: id,
        transformers.PreTrainedTokenizer: id
    }
)
def load_hooked_transformer(
    model_name: str,
    hf_model: Optional[transformers.PreTrainedModel] = None,
    tlens_device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    supported_model_name: Optional[str] = None,
):
    if model_name == "layerSkipCustom":
        tlens_model = load_custom_model("facebook/layerskip-llama3.2-1B")
        tlens_model.eval()
        return tlens_model
    
    if supported_model_name is None:
        supported_model_name = model_name
    supported_model_name = get_official_model_name(supported_model_name)
    if model_name not in MODEL_ALIASES:
        MODEL_ALIASES[supported_model_name] = []
    if model_name not in MODEL_ALIASES[supported_model_name]:
        MODEL_ALIASES[supported_model_name].append(model_name)
    
    tlens_model = transformer_lens.HookedTransformer.from_pretrained(
        model_name,
        hf_model=hf_model,
        fold_ln=False,  # Keep layer norm where it is.
        center_writing_weights=False,
        center_unembed=False,
        device=tlens_device,
        dtype=dtype,
    )
    tlens_model.eval()
    return tlens_model


# TODO(igortufanov): If we want to scale the app to multiple users, we need more careful
# thread-safe implementation. The simplest option could be to wrap the existing methods
# in mutexes.
class TransformerLensTransparentLlm(TransparentLlm):
    """
    Implementation of Transparent LLM based on transformer lens.

    Args:
    - model_name: The official name of the model from HuggingFace. Even if the model was
        patched or loaded locally, the name should still be official because that's how
        transformer_lens treats the model.
    - hf_model: The language model as a HuggingFace class.
    - tokenizer,
    - device: "gpu" or "cpu"
    """

    def __init__(
        self,
        model_name: str,
        hf_model: Optional[transformers.PreTrainedModel] = None,
        tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
        device: str = "gpu",
        dtype: torch.dtype = torch.float32,
        supported_model_name: str = None,
    ):
        if device == "gpu":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                RuntimeError("Asked to run on gpu, but torch couldn't find cuda or mps")
        elif device == "cpu":
            self.device = "cpu"
        else:
            raise RuntimeError(f"Specified device {device} is not a valid option")

        self.dtype = dtype
        self.hf_tokenizer = tokenizer
        self.hf_model = hf_model
        self._custom_model = None

        # self._model = tlens_model
        self._model_name = model_name
        self._supported_model_name = supported_model_name
        self._prepend_bos = True
        self._last_run = None
        self._run_exception = RuntimeError(
            "Tried to use the model output before calling the `run` method"
        )

        if model_name == "layerSkipCustom":
            self._custom_model = load_custom_model("facebook/layerskip-llama3.2-1B")

    def copy(self):
        import copy
        return copy.copy(self)

    @property
    def _model(self):
        tlens_model = load_hooked_transformer(
            self._model_name,
            hf_model=self.hf_model,
            tlens_device=self.device,
            dtype=self.dtype,
            supported_model_name=self._supported_model_name,
        )

        # if self._model_name == "layerSkipCustom":
        self._custom_model = tlens_model

        if self.hf_tokenizer is not None:
            tlens_model.set_tokenizer(self.hf_tokenizer, default_padding_side="left")

        if self._custom_model is None:
            tlens_model.set_use_attn_result(True)
            tlens_model.set_use_attn_in(False)
            tlens_model.set_use_split_qkv_input(False)

        return tlens_model

    def model_info(self) -> ModelInfo:
        if self._custom_model is not None:
            return self._custom_model.model_info()
        cfg = self._model.cfg
        return ModelInfo(
            name=self._model_name,
            n_params_estimate=cfg.n_params,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            d_model=cfg.d_model,
            d_vocab=cfg.d_vocab,
        )

    @torch.no_grad()
    def run(self, sentences: List[str]) -> None:
        if self._custom_model is not None:
            return self._custom_model.run(sentences)
        tokens = self._model.to_tokens(sentences, prepend_bos=self._prepend_bos)
        logits, cache = self._model.run_with_cache(tokens)

        self._last_run = _RunInfo(
            tokens=tokens,
            logits=logits,
            cache=cache,
        )

    def batch_size(self) -> int:
        if self._custom_model is not None:
            return self._custom_model.batch_size()
        if not self._last_run:
            raise self._run_exception
        return self._last_run.logits.shape[0]

    @typechecked
    def tokens(self) -> Int[torch.Tensor, "batch pos"]:
        if self._custom_model is not None:
            return self._custom_model.tokens()
        if not self._last_run:
            raise self._run_exception
        return self._last_run.tokens

    @typechecked
    def tokens_to_strings(self, tokens: Int[torch.Tensor, "pos"]) -> List[str]:
        if self._custom_model is not None:
            return self._custom_model.tokens_to_strings(tokens)
        return self._model.to_str_tokens(tokens)

    @typechecked
    def logits(self) -> Float[torch.Tensor, "batch pos d_vocab"]:
        if self._custom_model is not None:
            return self._custom_model.logits()
        if not self._last_run:
            raise self._run_exception
        return self._last_run.logits

    @torch.no_grad()
    @typechecked
    def unembed(
        self,
        t: Float[torch.Tensor, "d_model"],
        normalize: bool,
    ) -> Float[torch.Tensor, "vocab"]:
        if self._custom_model is not None:
            return self._custom_model.unembed(t, normalize)
        # t: [d_model] -> [batch, pos, d_model]
        tdim = t.unsqueeze(0).unsqueeze(0)
        if normalize:
            normalized = self._model.ln_final(tdim)
            result = self._model.unembed(normalized)
        else:
            result = self._model.unembed(tdim.to(self.dtype))
        return result[0][0]

    def _get_block(self, layer: int, block_name: str) -> torch.Tensor:
        if not self._last_run:
            raise self._run_exception
        return self._last_run.cache[f"blocks.{layer}.{block_name}"]

    # ================= Methods related to the residual stream =================

    @typechecked
    def residual_in(self, layer: int) -> Float[torch.Tensor, "batch pos d_model"]:
        if self._custom_model is not None:
            return self._custom_model.residual_in(layer)
        if not self._last_run:
            raise self._run_exception
        return self._get_block(layer, "hook_resid_pre")

    @typechecked
    def residual_after_attn(
        self, layer: int
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        if self._custom_model is not None:
            return self._custom_model.residual_after_attn(layer)
        if not self._last_run:
            raise self._run_exception
        return self._get_block(layer, "hook_resid_mid")

    @typechecked
    def residual_out(self, layer: int) -> Float[torch.Tensor, "batch pos d_model"]:
        if self._custom_model is not None:
            return self._custom_model.residual_out(layer)
        if not self._last_run:
            raise self._run_exception
        return self._get_block(layer, "hook_resid_post")

    # ================ Methods related to the feed-forward layer ===============

    @typechecked
    def ffn_out(self, layer: int) -> Float[torch.Tensor, "batch pos d_model"]:
        if self._custom_model is not None:
            return self._custom_model.ffn_out(layer)
        if not self._last_run:
            raise self._run_exception
        return self._get_block(layer, "hook_mlp_out")

    @torch.no_grad()
    @typechecked
    def decomposed_ffn_out(
        self,
        batch_i: int,
        layer: int,
        pos: int,
    ) -> Float[torch.Tensor, "hidden d_model"]:
        if self._custom_model is not None:
            return self._custom_model.decomposed_ffn_out(batch_i, layer, pos)
        # Take activations right before they're multiplied by W_out, i.e. non-linearity
        # and layer norm are already applied.
        processed_activations = self._get_block(layer, "mlp.hook_post")[batch_i][pos]
        return torch.mul(processed_activations.unsqueeze(-1), self._model.blocks[layer].mlp.W_out)

    @typechecked
    def neuron_activations(
        self,
        batch_i: int,
        layer: int,
        pos: int,
    ) -> Float[torch.Tensor, "hidden"]:
        if self._custom_model is not None:
            return self._custom_model.neuron_activations(batch_i, layer, pos)
        return self._get_block(layer, "mlp.hook_pre")[batch_i][pos]

    @typechecked
    def neuron_output(
        self,
        layer: int,
        neuron: int,
    ) -> Float[torch.Tensor, "d_model"]:
        if self._custom_model is not None:
            return self._custom_model.neuron_output(layer, neuron)
        return self._model.blocks[layer].mlp.W_out[neuron]

    # ==================== Methods related to the attention ====================

    @typechecked
    def attention_matrix(
        self, batch_i: int, layer: int, head: int
    ) -> Float[torch.Tensor, "query_pos key_pos"]:
        if self._custom_model is not None:
            return self._custom_model.attention_matrix(batch_i, layer, head)
        return self._get_block(layer, "attn.hook_pattern")[batch_i][head]

    @typechecked
    def attention_output_per_head(
        self,
        batch_i: int,
        layer: int,
        pos: int,
        head: int,
    ) -> Float[torch.Tensor, "d_model"]:
        if self._custom_model is not None:
            return self._custom_model.attention_output_per_head(batch_i, layer, pos, head)
        return self._get_block(layer, "attn.hook_result")[batch_i][pos][head]

    @typechecked
    def attention_output(
        self,
        batch_i: int,
        layer: int,
        pos: int,
    ) -> Float[torch.Tensor, "d_model"]:
        if self._custom_model is not None:
            return self._custom_model.attention_output(batch_i, layer, pos)
        return self._get_block(layer, "hook_attn_out")[batch_i][pos]

    @torch.no_grad()
    @typechecked
    def decomposed_attn(
        self, batch_i: int, layer: int
    ) -> Float[torch.Tensor, "pos key_pos head d_model"]:
        if self._custom_model is not None:
            return self._custom_model.decomposed_attn(batch_i, layer)
        if not self._last_run:
            raise self._run_exception
        hook_v = self._get_block(layer, "attn.hook_v")[batch_i]
        b_v = self._model.blocks[layer].attn.b_V

        # support for gqa
        num_head_groups = b_v.shape[-2] // hook_v.shape[-2]
        hook_v = hook_v.repeat_interleave(num_head_groups, dim=-2)

        v = hook_v + b_v
        pattern = self._get_block(layer, "attn.hook_pattern")[batch_i].to(v.dtype)
        z = einsum(
            "key_pos head d_head, "
            "head query_pos key_pos -> "
            "query_pos key_pos head d_head",
            v,
            pattern,
        )
        decomposed_attn = einsum(
            "pos key_pos head d_head, "
            "head d_head d_model -> "
            "pos key_pos head d_model",
            z,
            self._model.blocks[layer].attn.W_O,
        )
        return decomposed_attn
