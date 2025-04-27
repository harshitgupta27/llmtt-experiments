from dataclasses import dataclass
from typing import List, Optional

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from fancy_einsum import einsum
from jaxtyping import Float, Int
from typeguard import typechecked
import streamlit as st

from llm_transparency_tool.models.transparent_llm import ModelInfo, TransparentLlm

@dataclass
class _RunInfo:
    tokens: Int[torch.Tensor, "batch pos"]
    logits: Float[torch.Tensor, "batch pos d_vocab"]
    cache: dict  # Will store activations


@st.cache_resource(
    max_entries=1,
    show_spinner=True,
    hash_funcs={
        transformers.PreTrainedModel: id,
        transformers.PreTrainedTokenizer: id
    }
)
def load_layerskip_model(
    model_name: str,
    device: str = "auto",
    dtype: torch.dtype = torch.bfloat16,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map=device, 
        use_safetensors=True, 
        torch_dtype=dtype,
        attn_implementation="eager",
        output_hidden_states=True,
        output_attentions=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


class LayerSkipTransparentLlm(TransparentLlm):
    """
    Implementation of Transparent LLM for LayerSkip models.

    Args:
    - model_name: The official name of the model from HuggingFace.
    - device: "auto", "gpu" or "cpu"
    - dtype: The desired dtype for the model
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
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
        elif device == "auto":
            self.device = "auto"
        else:
            raise RuntimeError(f"Specified device {device} is not a valid option")

        self.dtype = dtype
        self._model_name = model_name
        self._prepend_bos = True
        self._last_run = None
        self._run_exception = RuntimeError(
            "Tried to use the model output before calling the `run` method"
        )
        
        # Load the model and tokenizer
        self.model, self.tokenizer = load_layerskip_model("facebook/layerskip-llama3.2-1B", self.device, self.dtype)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        # Cache for storing activations during forward pass
        self._activation_cache = {}
        
        # Register hooks to capture activations
        self._register_hooks()

    def copy(self):
        import copy
        return copy.copy(self)

    def _register_hooks(self):
        """Register forward hooks to capture model activations."""
        self._hooks = []
        
        # Helper function to create hook for capturing outputs
        def create_output_hook(layer_idx, hook_type):
            def hook_fn(module, input, output):
                self._activation_cache[f"blocks.{layer_idx}.{hook_type}"] = output
                return output
            return hook_fn
            
        # Helper function to create hook for capturing inputs
        def create_input_hook(layer_idx, hook_type):
            def hook_fn(module, input, output):
                # Input is typically a tuple, we want the first element
                if isinstance(input, tuple) and len(input) > 0:
                    self._activation_cache[f"blocks.{layer_idx}.{hook_type}"] = input[0]
                else:
                    self._activation_cache[f"blocks.{layer_idx}.{hook_type}"] = input
                return output
            return hook_fn
            
        # Register hooks for each layer
        n_layers = self.model.config.num_hidden_layers
        for layer_idx in range(n_layers):
            layer = self.model.model.layers[layer_idx]
            
            # Hooks for residual connections
            # Input to the layer (pre-layer norm)
            self._hooks.append(layer.register_forward_hook(
                create_input_hook(layer_idx, "hook_resid_pre")))
            
            # After attention but before MLP
            if hasattr(layer, "post_attention_layernorm"):
                self._hooks.append(layer.post_attention_layernorm.register_forward_hook(
                    create_input_hook(layer_idx, "hook_resid_mid")))
            
            # Output of the layer
            self._hooks.append(layer.register_forward_hook(
                create_output_hook(layer_idx, "hook_resid_post")))
            
            # Attention hooks
            self._hooks.append(layer.self_attn.register_forward_hook(
                create_output_hook(layer_idx, "hook_attn_out")))
            
            # More granular attention hooks
            self._hooks.append(layer.self_attn.q_proj.register_forward_hook(
                create_output_hook(layer_idx, "attn.hook_q")))
            self._hooks.append(layer.self_attn.k_proj.register_forward_hook(
                create_output_hook(layer_idx, "attn.hook_k")))
            self._hooks.append(layer.self_attn.v_proj.register_forward_hook(
                create_output_hook(layer_idx, "attn.hook_v")))
            
            # MLP hooks
            self._hooks.append(layer.mlp.register_forward_hook(
                create_output_hook(layer_idx, "hook_mlp_out")))
            
            # MLP component hooks
            self._hooks.append(layer.mlp.gate_proj.register_forward_hook(
                create_output_hook(layer_idx, "mlp.hook_pre")))
            
            if hasattr(layer.mlp, "up_proj"):
                self._hooks.append(layer.mlp.up_proj.register_forward_hook(
                    create_output_hook(layer_idx, "mlp.hook_up")))
            
            # Hook for the activation function output (if accessible)
            if hasattr(layer.mlp, "act_fn"):
                self._hooks.append(layer.mlp.act_fn.register_forward_hook(
                    create_output_hook(layer_idx, "mlp.hook_post")))
            
        # Add custom hook for capturing attention patterns - these may not be directly accessible
        def get_attention_pattern(module, input, output):
            # This needs to be implemented based on the specific model architecture
            # For now, this is a placeholder
            pass
            
        # Add hooks for other key components as needed
        
        # Add a hook for final model output before lm_head
        if hasattr(self.model.model, "norm"):
            self._hooks.append(self.model.model.norm.register_forward_hook(
                lambda m, i, o: self._activation_cache.update({"final_norm_output": o})))
    
    def model_info(self) -> ModelInfo:
        config = self.model.config
        # Estimate parameters based on model config
        n_params = config.num_hidden_layers * (
            # Attention parameters
            (4 * config.hidden_size * config.hidden_size) + 
            # MLP parameters
            (4 * config.hidden_size * config.intermediate_size) +
            # Other parameters (rough estimate)
            config.hidden_size * 10
        )
        
        return ModelInfo(
            name=self._model_name,
            n_params_estimate=n_params,
            n_layers=config.num_hidden_layers,
            n_heads=config.num_attention_heads,
            d_model=config.hidden_size,
            d_vocab=config.vocab_size,
        )

    @torch.no_grad()
    def run(self, sentences: List[str]) -> None:
        """Run the model on a batch of sentences and capture activations."""
        # Clear the cache
        self._activation_cache = {}
        
        # Tokenize input
        inputs = self.tokenizer(
            sentences, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            add_special_tokens=self._prepend_bos
        ).to(self.model.device)
        
        # Run model with output attentions and hidden states
        outputs = self.model(
            **inputs,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Store tokens and logits
        self._last_run = _RunInfo(
            tokens=inputs["input_ids"],
            logits=outputs.logits,
            cache=self._activation_cache
        )
        
        # Process and store attention patterns from outputs
        if outputs.attentions:
            for layer_idx, layer_attention in enumerate(outputs.attentions):
                self._activation_cache[f"blocks.{layer_idx}.attn.hook_pattern"] = layer_attention
                
                # Calculate attention result if not captured in hooks
                if f"blocks.{layer_idx}.attn.hook_result" not in self._activation_cache:
                    # This would require explicit calculation based on model architecture
                    # It's a placeholder for now
                    pass
        
        # Store hidden states from outputs if not already captured by hooks
        if outputs.hidden_states and len(outputs.hidden_states) > self.model.config.num_hidden_layers:
            for layer_idx in range(self.model.config.num_hidden_layers):
                # Only store if not already captured by hooks
                if f"blocks.{layer_idx}.hook_resid_pre" not in self._activation_cache:
                    self._activation_cache[f"blocks.{layer_idx}.hook_resid_pre"] = outputs.hidden_states[layer_idx]
                    
                if f"blocks.{layer_idx}.hook_resid_post" not in self._activation_cache:
                    self._activation_cache[f"blocks.{layer_idx}.hook_resid_post"] = outputs.hidden_states[layer_idx + 1]
        
        # Post-process activations if needed
        self._post_process_activations()

    def _post_process_activations(self):
        """Process activations that may need additional computation."""
        # Calculate any missing activations based on available ones
        n_layers = self.model.config.num_hidden_layers
        
        for layer_idx in range(n_layers):
            # Calculate residual mid if not captured directly
            if (f"blocks.{layer_idx}.hook_resid_mid" not in self._activation_cache and
                f"blocks.{layer_idx}.hook_resid_pre" in self._activation_cache and
                f"blocks.{layer_idx}.hook_attn_out" in self._activation_cache):
                
                resid_pre = self._activation_cache[f"blocks.{layer_idx}.hook_resid_pre"]
                attn_out = self._activation_cache[f"blocks.{layer_idx}.hook_attn_out"]
                
                # For many models, residual_mid = residual_pre + attn_out
                # But this depends on the specific architecture
                self._activation_cache[f"blocks.{layer_idx}.hook_resid_mid"] = resid_pre + attn_out
            
            # Calculate MLP post activations if not captured directly
            if (f"blocks.{layer_idx}.mlp.hook_post" not in self._activation_cache and
                f"blocks.{layer_idx}.mlp.hook_pre" in self._activation_cache and
                f"blocks.{layer_idx}.mlp.hook_up" in self._activation_cache):
                
                gate_output = self._activation_cache[f"blocks.{layer_idx}.mlp.hook_pre"]
                up_output = self._activation_cache[f"blocks.{layer_idx}.mlp.hook_up"]
                
                # SwiGLU activation calculation: gate_output * silu(up_output)
                # This is common in many modern models but check specific architecture
                silu = torch.nn.functional.silu(up_output)
                self._activation_cache[f"blocks.{layer_idx}.mlp.hook_post"] = gate_output * silu

    def batch_size(self) -> int:
        if not self._last_run:
            raise self._run_exception
        return self._last_run.logits.shape[0]

    @typechecked
    def tokens(self) -> Int[torch.Tensor, "batch pos"]:
        if not self._last_run:
            raise self._run_exception
        return self._last_run.tokens

    @typechecked
    def tokens_to_strings(self, tokens: Int[torch.Tensor, "pos"]) -> List[str]:
        tokens_list = tokens.tolist()
        return self.tokenizer.convert_ids_to_tokens(tokens_list)

    @typechecked
    def logits(self) -> Float[torch.Tensor, "batch pos d_vocab"]:
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
        # Expand dimensions for batch and position
        tdim = t.unsqueeze(0).unsqueeze(0)
        
        if normalize:
            # Apply layer norm if available
            if hasattr(self.model.model, "norm"):
                normalized = self.model.model.norm(tdim)
            else:
                normalized = tdim  # Fallback if no LN
        else:
            normalized = tdim
        
        # Apply unembedding (lm_head)
        result = self.model.lm_head(normalized)
        return result[0][0]

    def _get_block(self, layer: int, block_name: str) -> torch.Tensor:
        if not self._last_run:
            raise self._run_exception
        key = f"blocks.{layer}.{block_name}"
        if key not in self._activation_cache:
            raise KeyError(f"Activation for {key} not found in cache")
        return self._activation_cache[key]

    # ================= Methods related to the residual stream =================

    @typechecked
    def residual_in(self, layer: int) -> Float[torch.Tensor, "batch pos d_model"]:
        if not self._last_run:
            raise self._run_exception
        return self._get_block(layer, "hook_resid_pre")

    @typechecked
    def residual_after_attn(
        self, layer: int
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        if not self._last_run:
            raise self._run_exception
        return self._get_block(layer, "hook_resid_mid")

    @typechecked
    def residual_out(self, layer: int) -> Float[torch.Tensor, "batch pos d_model"]:
        if not self._last_run:
            raise self._run_exception
        return self._get_block(layer, "hook_resid_post")

    # ================ Methods related to the feed-forward layer ===============

    @typechecked
    def ffn_out(self, layer: int) -> Float[torch.Tensor, "batch pos d_model"]:
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
        # Get the processed activations before they're multiplied by W_out
        processed_activations = self._get_block(layer, "mlp.hook_post")[batch_i][pos]
        
        # Get the output projection matrix (W_out)
        # For LayerSkip models, this might be stored as down_proj
        layer_module = self.model.model.layers[layer].mlp
        w_out = layer_module.down_proj.weight.t()  # Transpose to match expected shape
        
        return torch.mul(processed_activations.unsqueeze(-1), w_out)

    @typechecked
    def neuron_activations(
        self,
        batch_i: int,
        layer: int,
        pos: int,
    ) -> Float[torch.Tensor, "hidden"]:
        return self._get_block(layer, "mlp.hook_pre")[batch_i][pos]

    @typechecked
    def neuron_output(
        self,
        layer: int,
        neuron: int,
    ) -> Float[torch.Tensor, "d_model"]:
        # For LayerSkip models, this might be the down_proj weights
        return self.model.model.layers[layer].mlp.down_proj.weight.t()[neuron]

    # ==================== Methods related to the attention ====================

    @typechecked
    def attention_matrix(
        self, batch_i: int, layer: int, head: int
    ) -> Float[torch.Tensor, "query_pos key_pos"]:
        return self._get_block(layer, "attn.hook_pattern")[batch_i][head]

    @typechecked
    def attention_output_per_head(
        self,
        batch_i: int,
        layer: int,
        pos: int,
        head: int,
    ) -> Float[torch.Tensor, "d_model"]:
        # Try to access hook_result if available
        try:
            return self._get_block(layer, "attn.hook_result")[batch_i][pos][head]
        except KeyError:
            # If hook_result isn't available, calculate it from other activations
            # This is a simplified calculation and may need to be adapted
            try:
                v = self._get_block(layer, "attn.hook_v")[batch_i]
                pattern = self._get_block(layer, "attn.hook_pattern")[batch_i]
                
                # Extract for specific position and head
                head_pattern = pattern[head][pos].unsqueeze(0)  # [1, seq_len]
                head_v = v[:, head, :]  # [seq_len, head_dim]
                
                # Calculate weighted value
                weighted_v = torch.matmul(head_pattern, head_v)  # [1, head_dim]
                
                # Get output projection
                layer_module = self.model.model.layers[layer].self_attn
                head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
                w_o = layer_module.o_proj.weight.view(
                    self.model.config.hidden_size,
                    self.model.config.num_attention_heads,
                    head_dim
                )[:, head, :]  # [hidden_size, head_dim]
                
                # Apply output projection
                return torch.matmul(weighted_v, w_o.t()).squeeze(0)  # [hidden_size]
                
            except Exception as e:
                raise NotImplementedError(f"attention_output_per_head calculation failed: {e}")

    @typechecked
    def attention_output(
        self,
        batch_i: int,
        layer: int,
        pos: int,
    ) -> Float[torch.Tensor, "d_model"]:
        return self._get_block(layer, "hook_attn_out")[batch_i][pos]

    @torch.no_grad()
    @typechecked
    def decomposed_attn(
        self, batch_i: int, layer: int
    ) -> Float[torch.Tensor, "pos key_pos head d_model"]:
        hook_v = self._get_block(layer, "attn.hook_v")[batch_i]
        
        # Get attention bias if available
        layer_module = self.model.model.layers[layer].self_attn
        if hasattr(layer_module, "v_proj") and hasattr(layer_module.v_proj, "bias") and layer_module.v_proj.bias is not None:
            b_v = layer_module.v_proj.bias
        else:
            b_v = 0  # No bias
            
        # Add bias to v if it exists
        v = hook_v + b_v
        
        # Get attention pattern
        pattern = self._get_block(layer, "attn.hook_pattern")[batch_i].to(v.dtype)
        
        # Calculate weighted values
        z = einsum(
            "key_pos head d_head, "
            "head query_pos key_pos -> "
            "query_pos key_pos head d_head",
            v,
            pattern,
        )
        
        # Get output projection matrix
        head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        o_proj = layer_module.o_proj.weight.view(
            self.model.config.hidden_size,
            self.model.config.num_attention_heads,
            head_dim
        ).permute(1, 2, 0)  # [head, d_head, d_model]
        
        # Apply output projection
        decomposed_attn = einsum(
            "pos key_pos head d_head, "
            "head d_head d_model -> "
            "pos key_pos head d_model",
            z,
            o_proj,
        )
        
        return decomposed_attn
        
    def remove_hooks(self):
        """Remove all hooks to avoid memory leaks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []