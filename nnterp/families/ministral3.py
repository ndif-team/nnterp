"""Ministral 3 (``Ministral3ForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward.
The queries are scaled by position (``llama_4_scaling_beta``) after the rotary
embedding and before the interface, so ``attention_queries`` are the scaled ones.
"""

from transformers.models.ministral3.modeling_ministral3 import (
    Ministral3Attention,
    Ministral3DecoderLayer,
    Ministral3MLP,
)

from ..components import Attention, Layer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """Ministral 3's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Ministral 3's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """Ministral 3's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Ministral3DecoderLayer: Layer, Ministral3Attention: Attention, Ministral3MLP: Mlp}
