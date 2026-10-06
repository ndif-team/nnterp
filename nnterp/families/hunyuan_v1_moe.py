"""Hunyuan MoE V1 (Hunyuan-A13B) (``HunYuanMoEV1ForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward.
The attention norms its queries and keys per head after the rotary embedding,
before the interface. Every block's MLP is a mixture of experts that adds a
shared expert (``mlp.shared_mlp``) to the routed sum and returns one tensor.
"""

from transformers.models.hunyuan_v1_moe.modeling_hunyuan_v1_moe import (
    HunYuanMoEV1Attention,
    HunYuanMoEV1DecoderLayer,
    HunYuanMoEV1Moe,
)

from ..components import Attention, Layer, Moe, RouterLogits, TokenEProperty

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "gate": "router",
    "shared_mlp": "shared_experts",
}


class Layer(Layer):
    """Hunyuan MoE V1's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Hunyuan MoE V1's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Moe):
    """A mixture of experts plus a shared expert (``shared_mlp``, run first), returned as one bare tensor; the residual is added in the block.

    The router (``gate``) projects with a child ``nn.Linear``, ``wg``, in float32:
    its output is the logits.
    """

    @TokenEProperty("router.wg.output", description=Moe.router_logits.description)
    def router_logits(self, value) -> RouterLogits:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {HunYuanMoEV1DecoderLayer: Layer, HunYuanMoEV1Attention: Attention, HunYuanMoEV1Moe: Mlp}
