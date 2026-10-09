"""GLM-4.7-Flash (``Glm4MoeLiteForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, which returns a bare tensor. Attention is DeepSeek-V2's
multi-head latent attention through the shared eager forward: queries and keys are
``qk_head_dim`` wide (``qk_nope_head_dim + qk_rope_head_dim``), values
``v_head_dim``, and the interface sees ``num_heads`` key/value heads. The config
maps ``head_dim`` to ``qk_rope_head_dim``, the rotary part only, so both widths come
from ``deepseek_v2``'s size functions.

``mlp_layer_types`` says which blocks are dense (the first, by default) and which
are a mixture of experts. The mixture returns the routed experts' sum plus a shared
expert's output as one tensor, so ``mlp_output`` is everything the block adds after
attention; the shared expert (``mlp.shared_experts``) is a dense MLP of the same class.
"""

from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
    Glm4MoeLiteAttention,
    Glm4MoeLiteDecoderLayer,
    Glm4MoeLiteMLP,
    Glm4MoeLiteMoE,
)

from ..components import Attention, Layer, Mlp, Moe
from .deepseek_v2 import head_dim, qk_head_dim  # noqa: F401  the same latent attention: the same sizes

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "gate": "router",
}


class Layer(Layer):
    """GLM-4-MoE-Lite's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """GLM-4-MoE-Lite's latent attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """GLM-4-MoE-Lite's dense MLP or mixture of experts; both return the hidden states (the mixture with its shared expert added), and the residual is added in the block."""


class Moe(Moe, Mlp):
    """GLM-4-MoE-Lite's mixture of experts: DeepSeek-V3's sigmoid router, routed experts and a shared expert."""

    SCORING = "sigmoid"


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Glm4MoeLiteDecoderLayer: Layer, Glm4MoeLiteAttention: Attention, Glm4MoeLiteMLP: Mlp, Glm4MoeLiteMoE: Moe}
