"""ERNIE 4.5 dense (``Ernie4_5ForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward.
"""

from transformers.models.ernie4_5.modeling_ernie4_5 import Ernie4_5Attention, Ernie4_5DecoderLayer, Ernie4_5MLP

from ..components import Attention, Layer, Mlp

MODEL_TYPES = ("ernie4_5",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """ERNIE 4.5 dense's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """ERNIE 4.5 dense's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """ERNIE 4.5 dense's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Ernie4_5DecoderLayer: Layer, Ernie4_5Attention: Attention, Ernie4_5MLP: Mlp}
