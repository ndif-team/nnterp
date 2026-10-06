"""Llama (``LlamaForCausalLM``), the family the standard vocabulary is taken from.

Block names (``input_layernorm``, ``self_attn``, ``post_attention_layernorm``,
``mlp``) and ``lm_head`` are already the standard ones. The only change is
lifting the containers out of ``model.model``: ``model.layers`` instead of
``model.model.layers``.
"""

from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer, LlamaMLP

from ..components import Attention, Layer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """Llama's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Llama's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """Llama's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {LlamaDecoderLayer: Layer, LlamaAttention: Attention, LlamaMLP: Mlp}
