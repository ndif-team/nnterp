"""Phi 1 / 1.5 / 2 (``PhiForCausalLM``).

Llama's containers with a parallel block: one ``input_layernorm`` feeds both
sublayers and ``x + attn + mlp`` is summed at the end (``resid_dropout`` on
each, inert in eval). The final norm is ``final_layernorm``. There is no
``post_attention_layernorm``.
"""

from transformers.models.phi.modeling_phi import PhiAttention, PhiDecoderLayer, PhiMLP

from ..components import Attention, Layer, Mlp

MODEL_TYPES = ("phi",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.final_layernorm": "norm",
}


class Layer(Layer):
    """Phi's parallel block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Phi's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """Phi's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {PhiDecoderLayer: Layer, PhiAttention: Attention, PhiMLP: Mlp}
