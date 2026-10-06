"""Ministral (``MinistralForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward.
Every block may take a sliding window (``layer_types``); the window is a mask
given to the interface, so the values are read as it applies.
"""

from transformers.models.ministral.modeling_ministral import MinistralAttention, MinistralDecoderLayer, MinistralMLP

from ..components import Attention, Layer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """Ministral's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Ministral's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """Ministral's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {MinistralDecoderLayer: Layer, MinistralAttention: Attention, MinistralMLP: Mlp}
