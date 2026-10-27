"""OLMo 1 (``OlmoForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward.
"""

from transformers.models.olmo.modeling_olmo import OlmoAttention, OlmoDecoderLayer, OlmoMLP

from ..components import Attention, Layer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """OLMo's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """OLMo's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """OLMo's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {OlmoDecoderLayer: Layer, OlmoAttention: Attention, OlmoMLP: Mlp}
