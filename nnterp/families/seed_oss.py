"""Seed-OSS (``SeedOssForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward.
The attention output and the MLP output each pass a residual dropout inside
their module, which is the identity in eval.
"""

from transformers.models.seed_oss.modeling_seed_oss import SeedOssAttention, SeedOssDecoderLayer, SeedOssMLP

from ..components import Attention, Layer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """Seed-OSS's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Seed-OSS's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """Seed-OSS's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {SeedOssDecoderLayer: Layer, SeedOssAttention: Attention, SeedOssMLP: Mlp}
