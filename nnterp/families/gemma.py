"""Gemma 1 (``GemmaForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward.
"""

from transformers.models.gemma.modeling_gemma import GemmaAttention, GemmaDecoderLayer, GemmaMLP

from ..components import Attention, Layer, Mlp

MODEL_TYPES = ("gemma",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """Gemma's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Gemma's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """Gemma's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {GemmaDecoderLayer: Layer, GemmaAttention: Attention, GemmaMLP: Mlp}
