import torch
import math
from typing import List
from jaxtyping import Float, Int
from llm_transparency_tool.models.transparent_llm import TransparentLlm, ModelInfo
from transformers import AutoModelForCausalLM, AutoTokenizer
from copy import deepcopy

class LayerSkipLlamaTransparentLlm(TransparentLlm):
    def __init__(self, model_name: str = "facebook/layerskip-llama3.2-1B"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", use_safetensors=True, torch_dtype=torch.bfloat16, attn_implementation="eager")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._last_run = None
        self._run_exception = RuntimeError("Tried to use the model output before calling the `run` method")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.output_hidden_states = True

    def eval(self):
        self.model.eval()
        return self
    
    def train(self, mode=True):
        self.model.train(mode)
        return self
    
    def model_info(self) -> ModelInfo:
        config = self.model.config
        return ModelInfo(
            name=config.name_or_path,
            n_params_estimate=sum(p.numel() for p in self.model.parameters()),
            n_layers=config.num_hidden_layers,
            n_heads=config.num_attention_heads,
            d_model=config.hidden_size,
            d_vocab=config.vocab_size
        )

    
    @torch.no_grad()
    def run(self, sentences: List[str]) -> None:
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, output_attentions=True)
        self.logits = outputs.logits
        self.cache = {
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
            'input_ids': inputs.input_ids
        }
        self._last_run = {
            'logits': self.logits,
            'input_ids': self.cache['input_ids'],
            'hidden_states': self.cache['hidden_states'],
            'attentions': self.cache['attentions']
        }

    def batch_size(self) -> int:
        if not self._last_run:
            raise self._run_exception
        return self._last_run['input_ids'].shape[0]

    def tokens(self) -> Int[torch.Tensor, "batch pos"]:
        if not self._last_run:
            raise self._run_exception
        return self._last_run['input_ids']

    def tokens_to_strings(self, tokens: Int[torch.Tensor, "pos"]) -> List[str]:
        return self.tokenizer.batch_decode(tokens)

    def logits(self) -> Float[torch.Tensor, "batch pos d_vocab"]:
        if not self._last_run:
            raise self._run_exception
        return self._last_run['logits']

    @torch.no_grad()
    def unembed(self, t: Float[torch.Tensor, "d_model"], normalize: bool) -> Float[torch.Tensor, "vocab"]:
        if normalize:
            t = self.model.model.norm(t)
        return self.model.lm_head(t.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

    def _get_hidden_state(self, layer: int) -> torch.Tensor:
        if not self._last_run:
            raise self._run_exception
        return self._last_run['hidden_states'][layer]

    def residual_in(self, layer: int) -> Float[torch.Tensor, "batch pos d_model"]:
        return self._get_hidden_state(layer)

    def residual_out(self, layer: int) -> Float[torch.Tensor, "batch pos d_model"]:
        return self._get_hidden_state(layer + 1)

    def residual_after_attn(self, layer: int) -> Float[torch.Tensor, "batch pos d_model"]:
        if layer >= len(self.cache['hidden_states']) - 1:
            raise ValueError(f"Layer {layer} is out of range")
    
        # Approximate the residual after attention as the average of the current and next layer's hidden states
        current_hidden = self.cache['hidden_states'][layer]
        next_hidden = self.cache['hidden_states'][layer + 1]
        return (current_hidden + next_hidden) / 2

    def attention_matrix(self, batch_i: int, layer: int, head: int) -> Float[torch.Tensor, "query_pos key_pos"]:
        if not self._last_run:
            raise self._run_exception
        return self._last_run['attentions'][layer][batch_i, head]

    # Placeholder methods for unimplemented features

    def ffn_out(self, layer: int) -> Float[torch.Tensor, "batch pos d_model"]:
        if not self.cache:
            raise RuntimeError("Model has not been run yet. Call run() first.")
        return self.cache['hidden_states'][layer + 1] - self.cache['hidden_states'][layer]
    
    def decomposed_ffn_out(
    self,
    batch_i: int,
    layer: int,
    pos: int,
    ) -> Float[torch.Tensor, "hidden d_model"]:
        if not self.cache:
            raise RuntimeError("Model has not been run yet. Call run() first.")
        
        ffn = self.model.model.layers[layer].mlp
        hidden_state = self.cache['hidden_states'][layer][batch_i, pos]
        
        # Get intermediate activations
        gate = torch.nn.functional.silu(ffn.gate_proj(hidden_state))
        up = ffn.up_proj(hidden_state)
        intermediate = gate * up  # Shape: [hidden_size]
        
        # Transpose weight matrix and adjust einsum dimensions
        return torch.einsum('i,ji->ij', intermediate, ffn.down_proj.weight)

    def neuron_activations(
        self,
        batch_i: int,
        layer: int,
        pos: int,
    ) -> Float[torch.Tensor, "d_ffn"]:
        if not self.cache:
            raise RuntimeError("Model has not been run yet. Call run() first.")
        ffn = self.model.model.layers[layer].mlp
        hidden_states = self.cache['hidden_states'][layer][batch_i, pos]
        gate_proj = ffn.gate_proj(hidden_states)
        up_proj = ffn.up_proj(hidden_states)
        return torch.nn.functional.silu(gate_proj) * up_proj

    def neuron_output(
        self,
        layer: int,
        neuron: int,
    ) -> Float[torch.Tensor, "d_model"]:
        return self.model.model.layers[layer].mlp.down_proj.weight[neuron]

    def attention_output(
        self,
        batch_i: int,
        layer: int,
        pos: int,
        head: int,
    ) -> Float[torch.Tensor, "d_model"]:
        if not self.cache:
            raise RuntimeError("Model has not been run yet. Call run() first.")
        attn_output = self.cache['attentions'][layer][batch_i, head, pos]
        return self.model.model.layers[layer].self_attn.o_proj(attn_output.unsqueeze(0)).squeeze(0)

    def attention_output_per_head(
        self,
        batch_i: int,
        layer: int,
        pos: int,
        head: int,
    ) -> Float[torch.Tensor, "d_model"]:
        if not self.cache:
            raise RuntimeError("Model has not been run yet. Call run() first.")
        # Clone the tensor to convert inference tensor to regular tensor
        attn_output = self.cache['attentions'][layer][batch_i, head, pos].clone()
        return self.model.model.layers[layer].self_attn.o_proj(attn_output.unsqueeze(0)).squeeze(0)

    def decomposed_attn(
        self, batch_i: int, layer: int
    ) -> Float[torch.Tensor, "pos key_pos head d_model"]:
        if not self.cache:
            raise RuntimeError("Model has not been run yet. Call run() first.")
        
        attn_layer = self.model.model.layers[layer].self_attn
        config = self.model.config
        
        # Get attention parameters
        num_heads = config.num_attention_heads
        num_key_value_heads = getattr(config, 'num_key_value_heads', num_heads)
        head_dim = config.hidden_size // num_heads
        
        hidden_states = self.cache['hidden_states'][layer][batch_i]
        
        # Project queries/keys/values
        q = attn_layer.q_proj(hidden_states)
        k = attn_layer.k_proj(hidden_states)
        v = attn_layer.v_proj(hidden_states)

        # Reshape tensors accounting for GQA
        q = q.view(-1, num_heads, head_dim)
        k = k.view(-1, num_key_value_heads, head_dim)
        v = v.view(-1, num_key_value_heads, head_dim)

        # Handle grouped query attention by repeating k/v heads
        if num_key_value_heads != num_heads:
            reps = num_heads // num_key_value_heads
            k = k[:, :, None, :].repeat(1, 1, reps, 1).flatten(1, 2)
            v = v[:, :, None, :].repeat(1, 1, reps, 1).flatten(1, 2)

        # Calculate attention weights with proper dimension alignment
        attn_weights = torch.einsum('qhd,khd->hqk', q, k) / math.sqrt(head_dim)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        
        # Apply attention to values
        attn_output = torch.einsum('hqk,khd->qkhd', attn_weights, v)
        
        # Project to output space
        return torch.einsum(
            'qkhd,hdm->qkhm',
            attn_output,
            attn_layer.o_proj.weight.view(num_heads, head_dim, config.hidden_size)
        )
