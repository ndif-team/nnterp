"""Kimi-Linear (``KimiLinearForCausalLM``).

Llama's tree and Llama's pre-norm block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, each sublayer's
output added to the stream in the block, a bare tensor returned. A hybrid by
``config.layer_types``: most blocks hold a Kimi Delta Attention mixer
(``KimiLinearDeltaAttention``, ``linear_attention``), the rest DeepSeek-V3's
multi-head latent attention without rotary embeddings (``KimiLinearAttention``,
``full_attention``; three KDA blocks to one MLA block on the released checkpoints,
whose last block is MLA too).

transformers keeps both mixers under the native name ``self_attn``. The standard name
follows the class, as on Nemotron-H: ``RENAME`` keys ``linear_attn`` on the KDA class,
and the family's `Layer` clears ``self_attn`` on a KDA block, so each block has
``self_attn`` or ``linear_attn``, never both, and a KDA block reports every
``self_attn`` value missing. The KDA module stays reachable as ``linear_attn``.

Kimi Delta Attention is a gated DeltaNet whose decay is per key channel: the forget
gate yields ``g`` of ``[batch, seq, heads, key_dim]`` (``ChannelGates``) where the
gated DeltaNet's is one per head, and the state decays channel by channel before each
delta-rule update. The forward is the gated DeltaNet's (the padding mask, the cached
branch, the conv, then a chunked kernel on a prompt and a token-by-token one on a
decode step, each taking ``(query, key, value, g=, beta=, initial_state=)`` and
returning ``(core_attn_out, last_recurrent_state)``), with transformers' own KDA
kernels, ``chunk_kimi_delta_attention`` / ``recurrent_kimi_delta_attention``, so
`LinearAttention` holds with those kernel names and the decay's layout. The queries
and keys reach the kernel before their L2 norm, which the kernel applies
(``use_qk_l2norm_in_kernel``).

The first ``first_k_dense_replace`` blocks (``mlp_layer_types``) have a dense MLP, the
rest DeepSeek-V3's mixture of experts (a sigmoid router with a selection bias, routed
experts and a shared expert). The attention sizes are DeepSeek-V2's.
"""

import torch
from transformers.models.kimi_linear.modeling_kimi_linear import (
    KimiLinearAttention,
    KimiLinearDecoderLayer,
    KimiLinearDeltaAttention,
    KimiLinearMLP,
    KimiLinearMoE,
)

from ..components import Attention, ChannelGates, EProperty, Layer, LinearAttention, Mlp, Moe, needs_torch_kernels
from ..components.recurrent import kernel
from .deepseek_v2 import head_dim, qk_head_dim  # noqa: F401  the same latent attention: the same sizes

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    KimiLinearDeltaAttention: "linear_attn",
    "gate": "router",
}


class Layer(Layer):
    """Kimi-Linear's decoder block; returns a bare tensor, so the base holds.

    On a KDA block the native ``self_attn`` is the KDA mixer, aliased
    ``linear_attn``; ``self_attn`` is cleared there, so it names the softmax
    attention only.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self._module.self_attn, KimiLinearDeltaAttention):
            object.__setattr__(self, "self_attn", None)


class Attention(Attention):
    """Kimi-Linear's latent attention (one block in four); DeepSeek-V3's without rotary, through the shared eager forward, so the base holds."""


class LinearAttention(LinearAttention):
    """Kimi Delta Attention: the gated DeltaNet's values at transformers' KDA kernels, with a decay per key channel."""

    CHUNK_KERNEL = "chunk_kimi_delta_attention_0"
    RECURRENT_KERNEL = "recurrent_kimi_delta_attention_0"
    #: The recurrent kernel binds the state once before its loop and twice per token: decay, then update.
    STATE_OP = "last_recurrent_state_2"

    @EProperty(kernel("inputs"), select="g", description="The per-token, per-channel log decay of the recurrent state", unavailable=needs_torch_kernels)
    def decays(self, value: torch.Tensor) -> ChannelGates:
        """The forget gate: the log of how much of each key channel of the state each token keeps, ``[batch, seq, heads, key_dim]``, float32 and non-positive."""
        return value


class Mlp(Mlp):
    """Kimi-Linear's dense MLP (also the shared expert); returns the hidden states, and the residual is added in the block."""


class Moe(Moe, Mlp):
    """Kimi-Linear's mixture of experts: DeepSeek-V3's, a sigmoid router with a selection bias, routed experts and a shared expert."""

    SCORING = "sigmoid"


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {
    KimiLinearDecoderLayer: Layer,
    KimiLinearAttention: Attention,
    KimiLinearDeltaAttention: LinearAttention,
    KimiLinearMLP: Mlp,
    KimiLinearMoE: Moe,
}
