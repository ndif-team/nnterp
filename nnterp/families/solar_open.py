"""Solar Open (``SolarOpenForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward.
The rotary covers ``partial_rotary_factor`` of each head (1.0, the whole head, on
Solar-Open-100B). Every block's MLP is a mixture of experts: transformers builds
``SolarOpenMoE`` on every block whatever ``first_k_dense_replace`` says. It adds a
shared expert (``mlp.shared_experts``, a ``SolarOpenMLP``) before it returns; the
experts are ``moe_intermediate_size`` wide. ``SolarOpenMLP`` is keyed to `Mlp`, so the
shared expert carries its own ``mlp_output``.
"""

from transformers.models.solar_open.modeling_solar_open import (
    SolarOpenAttention,
    SolarOpenDecoderLayer,
    SolarOpenMLP,
    SolarOpenMoE,
)

from ..components import Attention, Layer, Mlp, Moe

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "gate": "router",
}


class Layer(Layer):
    """Solar Open's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Solar Open's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """Solar Open's dense MLP or mixture of experts; both return the hidden states (the mixture with its shared expert added), and the residual is added in the block."""


class Moe(Moe, Mlp):
    """Solar Open's mixture of experts: DeepSeek-V3's sigmoid router, routed experts and a shared expert."""

    SCORING = "sigmoid"


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {SolarOpenDecoderLayer: Layer, SolarOpenAttention: Attention, SolarOpenMLP: Mlp, SolarOpenMoE: Moe}
