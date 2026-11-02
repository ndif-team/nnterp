"""Apertus on vLLM (``vllm.model_executor.models.apertus``).

Llama's names but for the norms, which are transformers' too: the block's
``attention_layernorm`` feeds the attention and ``feedforward_layernorm`` the
MLP, aliased ``input_layernorm`` and ``post_attention_layernorm``. The block
is fused: it takes and returns the stream as ``(hidden_states, residual)``.
The attention norms its queries and keys per head before the rotary
embedding, so ``attention_queries`` and ``attention_keys`` are the normed,
rotated ones, as on transformers. The MLP has no gate
(``down_proj(xielu(up_proj(x)))``).
"""

from vllm.model_executor.models.apertus import ApertusAttention, ApertusDecoderLayer, ApertusMLP

from ...components.vllm import Attention, FusedLayer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "attention_layernorm": "input_layernorm",
    "feedforward_layernorm": "post_attention_layernorm",
}


class Layer(FusedLayer):
    """The decoder block: ``forward(positions, hidden_states, residual) -> (hidden_states, residual)``."""


class Attention(Attention):
    """The attention; its output is what the block adds, so the base holds."""


class Mlp(Mlp):
    """The ungated MLP; its output is what the block adds, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {ApertusDecoderLayer: Layer, ApertusAttention: Attention, ApertusMLP: Mlp}
