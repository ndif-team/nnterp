"""MiMo-V2-Flash (``MiMoV2FlashForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward. Queries and keys
are ``head_dim`` wide and values ``v_head_dim`` wide, so the family defines
``qk_head_dim`` as the config's ``head_dim`` and ``head_dim`` as ``v_head_dim``. The
values are scaled by ``attention_value_scale`` before the interface.

``layer_types`` alternates full and sliding-window attention. A sliding block has
twice ``num_key_value_heads`` key/value heads and an **attention sink** per head: a
learned logit that joins the softmax as one extra key column and is dropped
afterwards, so that block's pattern rows sum to less than one. ``num_kv_heads`` is
the config's, the full blocks' count. The softmax input is shifted by its row max
(and on a sliding block is one key wider), so ``attention_scores`` is read one step
earlier, at the masked scores, on both block types. ``mlp_layer_types`` says which
blocks are dense and which a mixture of experts returning the routed hidden states.
"""

from typing import TYPE_CHECKING

from transformers.models.mimo_v2_flash.modeling_mimo_v2_flash import (
    MiMoV2FlashAttention,
    MiMoV2FlashDecoderLayer,
    MiMoV2FlashMLP,
    MiMoV2FlashMoE,
)

from ..components import Attention, EProperty, INTERFACE, Layer, Mlp, Moe, Pattern, interface_reason

if TYPE_CHECKING:
    from ..standardized import StandardizedTransformer

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "gate": "router",
}


class Layer(Layer):
    """MiMo-V2-Flash's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """MiMo-V2-Flash's attention: the shared interface, with a sink column in the softmax on sliding blocks.

    The pattern (the dropout output) is already without the sink column, so the
    base holds there; the scores are the masked scores, bound before the sink
    column joins and before the row max is subtracted.
    """

    #: The pattern's rows sum to less than one on the sliding-window blocks, where the sink takes the rest.
    SINK = True

    @EProperty(f"source.{INTERFACE}.source.attn_weights_1.output", description=Attention.attention_scores.description, unavailable=interface_reason)
    def attention_scores(self, value) -> Pattern:
        return value


class Mlp(Mlp):
    """MiMo-V2-Flash's dense MLP or mixture of experts; both return the hidden states, and the residual is added in the block."""


class Moe(Moe, Mlp):
    """MiMo-V2-Flash's mixture of experts: DeepSeek-V3's sigmoid router and routed experts, no shared expert."""

    SCORING = "sigmoid"


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {MiMoV2FlashDecoderLayer: Layer, MiMoV2FlashAttention: Attention, MiMoV2FlashMLP: Mlp, MiMoV2FlashMoE: Moe}


# -- sizes: queries and keys are head_dim wide, values v_head_dim ------------------

def head_dim(model: "StandardizedTransformer") -> int:
    """Width of one head's values and outputs: ``v_head_dim``."""
    return model.config.v_head_dim


def qk_head_dim(model: "StandardizedTransformer") -> int:
    """Width of one head's queries and keys: the config's ``head_dim``."""
    return model.config.head_dim
