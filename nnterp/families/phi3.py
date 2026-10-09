"""Phi-3 / 3.5 / 4 (``Phi3ForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward. Queries, keys and values are indexed out of one fused ``qkv_proj``.
"""

from transformers.models.phi3.modeling_phi3 import Phi3Attention, Phi3DecoderLayer, Phi3MLP

from ..components import Attention, Layer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """Phi-3's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Phi-3's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """Phi-3's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Phi3DecoderLayer: Layer, Phi3Attention: Attention, Phi3MLP: Mlp}
