"""Hunyuan dense V1 (``HunYuanDenseV1ForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward.
The attention norms its queries and keys per head (``query_layernorm``,
``key_layernorm``) after the rotary embedding and before the interface, so the
queries and keys are read after both.
"""

from transformers.models.hunyuan_v1_dense.modeling_hunyuan_v1_dense import (
    HunYuanDenseV1Attention,
    HunYuanDenseV1DecoderLayer,
    HunYuanDenseV1MLP,
)

from ..components import Attention, Layer, Mlp

MODEL_TYPES = ("hunyuan_v1_dense",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """Hunyuan dense V1's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Hunyuan dense V1's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """Hunyuan dense V1's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {HunYuanDenseV1DecoderLayer: Layer, HunYuanDenseV1Attention: Attention, HunYuanDenseV1MLP: Mlp}
