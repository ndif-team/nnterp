"""Llama on vLLM (``vllm.model_executor.models.llama``).

The names are transformers' (``model.layers[i].{input_layernorm, self_attn,
post_attention_layernorm, mlp}``, ``model.norm``, ``lm_head``). The block
fuses each residual add into the norm that follows, so it takes and returns
the stream as ``(hidden_states, residual)``.
"""

from vllm.model_executor.models.llama import LlamaAttention, LlamaDecoderLayer, LlamaMLP

from ...components.vllm import Attention, FusedLayer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(FusedLayer):
    """Llama's decoder block: ``forward(positions, hidden_states, residual) -> (hidden_states, residual)``."""


class Attention(Attention):
    """Llama's attention; its output is what the block adds, so the base holds."""


class Mlp(Mlp):
    """Llama's MLP; its output is what the block adds, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {LlamaDecoderLayer: Layer, LlamaAttention: Attention, LlamaMLP: Mlp}
