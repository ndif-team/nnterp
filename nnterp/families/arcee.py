"""Arcee (AFM) (``ArceeForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward.
The MLP has no gate: ``down_proj(act(up_proj(x)))`` with ReLU squared.
"""

from transformers.models.arcee.modeling_arcee import ArceeAttention, ArceeDecoderLayer, ArceeMLP

from ..components import Attention, Layer, Mlp

MODEL_TYPES = ("arcee",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """Arcee's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Arcee's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """Arcee's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {ArceeDecoderLayer: Layer, ArceeAttention: Attention, ArceeMLP: Mlp}
