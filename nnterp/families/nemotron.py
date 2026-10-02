"""Nemotron (Nemotron-4 / Minitron) (``NemotronForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward.
The norms are ``NemotronLayerNorm1P`` (a LayerNorm whose weight is stored
minus one); the rotary covers ``partial_rotary_factor`` of each head; the MLP
has no gate (``down_proj(relu2(up_proj(x)))``).
"""

from transformers.models.nemotron.modeling_nemotron import NemotronAttention, NemotronDecoderLayer, NemotronMLP

from ..components import Attention, Layer, Mlp

MODEL_TYPES = ("nemotron",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """Nemotron's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Nemotron's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """Nemotron's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {NemotronDecoderLayer: Layer, NemotronAttention: Attention, NemotronMLP: Mlp}
