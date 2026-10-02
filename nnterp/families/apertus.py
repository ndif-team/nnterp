"""Apertus (``ApertusForCausalLM``).

Llama's tree and Llama's block under other norm names: ``model.{embed_tokens,
layers[i].{attention_layernorm, self_attn, feedforward_layernorm, mlp}, norm}`` and
``lm_head``, the residual added in the block, attention through the shared eager
forward. ``attention_layernorm`` feeds the attention and ``feedforward_layernorm``
the MLP, so they are aliased ``input_layernorm`` and ``post_attention_layernorm``.
The attention norms its queries and keys per head (``q_norm``, ``k_norm``) before
the rotary embedding. The MLP has no gate: ``down_proj(xielu(up_proj(x)))``.
"""

from transformers.models.apertus.modeling_apertus import ApertusAttention, ApertusDecoderLayer, ApertusMLP

from ..components import Attention, Layer, Mlp

MODEL_TYPES = ("apertus",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "attention_layernorm": "input_layernorm",
    "feedforward_layernorm": "post_attention_layernorm",
}


class Layer(Layer):
    """Apertus's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Apertus's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """Apertus's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {ApertusDecoderLayer: Layer, ApertusAttention: Attention, ApertusMLP: Mlp}
