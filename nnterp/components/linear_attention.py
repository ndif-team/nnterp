"""`LinearAttention`: a gated DeltaNet mixer's values, at the delta-rule kernel call."""

from __future__ import annotations

from typing import Any

import torch

from jaxtyping import Float
from torch import Tensor

from .eproperty import EProperty
from .recurrent import RecurrentMixer, State, kernel, needs_torch_kernels
from .standard import BATCH, KEY_DIM, MIXER_HEADS, SEQ, VALUE_DIM

#: The layouts at the delta-rule call, tokens before heads: the queries and keys on the state's key side, the
#: values and each head's read of the state on its value side, and one gate per token and head. The state itself
#: (`State`, `States`) is the base's.
LinearQK = Float[Tensor, f"{BATCH} {SEQ} {MIXER_HEADS} {KEY_DIM}"]
LinearV = Float[Tensor, f"{BATCH} {SEQ} {MIXER_HEADS} {VALUE_DIM}"]
Gates = Float[Tensor, f"{BATCH} {SEQ} {MIXER_HEADS}"]
#: A gate per token, head and key channel: Kimi Delta Attention decays each key channel of the state on its own.
ChannelGates = Float[Tensor, f"{BATCH} {SEQ} {MIXER_HEADS} {KEY_DIM}"]


class LinearAttention(RecurrentMixer):
    """A gated DeltaNet mixer (Qwen3.5's ``linear_attn``): linear attention with a recurrent state.

    It projects queries, keys and values like attention, but mixes them
    through a per-head recurrent state instead of a softmax over keys: for
    each token the state decays by a learned gate (``decays``), takes up the
    new key/value pair scaled by ``betas``, and the query reads against it.
    So the shared names mean what they mean on `Attention`, and there is no
    pattern and no scores; instead there are the gate, the beta, and the
    state entering and leaving the layer, one ``key_dim`` by ``value_dim``
    matrix per head. On a decode step the sequence axis is 1 and
    ``state_input`` is the cached state.
    """

    #: The call a prompt runs through: ``torch_chunk_gated_delta_rule(query, key, value, g=, beta=, initial_state=, ...)``.
    CHUNK_KERNEL = "torch_chunk_gated_delta_rule_0"
    #: The call each decode step runs through, with the same arguments.
    RECURRENT_KERNEL = "torch_recurrent_gated_delta_rule_0"
    #: Inside the token-by-token kernel, the binding of the state after each token's update.
    STATE_OP = "last_recurrent_state_3"

    @EProperty(kernel("inputs"), select=0, description="The queries entering the delta rule", unavailable=needs_torch_kernels)
    def attention_queries(self, value: torch.Tensor) -> LinearQK:
        """The queries the delta-rule call receives, ``[batch, seq, heads, key_dim]``, after the conv and the activation: one per query/key head where the kernel repeats them to the value heads itself, one per value head where the mixer's forward has repeated them before the call."""
        return value

    @EProperty(kernel("inputs"), select=1, description="The keys entering the delta rule", unavailable=needs_torch_kernels)
    def attention_keys(self, value: torch.Tensor) -> LinearQK:
        """The keys the delta-rule call receives, ``[batch, seq, heads, key_dim]``, with as many heads as the queries."""
        return value

    @EProperty(kernel("inputs"), select=2, description="The values entering the delta rule", unavailable=needs_torch_kernels)
    def attention_values(self, value: torch.Tensor) -> LinearV:
        """The values the delta rule receives, ``[batch, seq, heads, value_dim]``."""
        return value

    @EProperty(kernel("inputs"), select="g", description="The per-token log decay of the recurrent state", unavailable=needs_torch_kernels)
    def decays(self, value: torch.Tensor) -> Gates:
        """The gate: the log of how much of the state each token keeps, ``[batch, seq, heads]``, float32 and non-positive."""
        return value

    @EProperty(kernel("inputs"), select="beta", description="The per-token write strength into the state", unavailable=needs_torch_kernels)
    def betas(self, value: torch.Tensor) -> Gates:
        """How strongly each token's key/value pair is written into the state, ``[batch, seq, heads]``, in ``(0, 1)`` (``(0, 2)`` where the config sets ``linear_allow_neg_eigval``)."""
        return value

    @EProperty(kernel("inputs"), select="initial_state", description="The recurrent state entering the layer, or None at the start of a prompt (a copy of the cache's buffer)", unavailable=needs_torch_kernels)
    def state_input(self, value: Any) -> State | None:
        """The state this call starts from: ``None`` on a fresh prompt, the cached state on a decode step.

        A copy, since the cache overwrites its buffer in place with the step's
        new state. Assign to replace what the step starts from.
        """
        return value if value is None else value.clone()

    @EProperty(kernel("output"), select=0, description="The per-head outputs before the gated norm and the output projection", unavailable=needs_torch_kernels)
    def attention_head_outputs(self, value: torch.Tensor) -> LinearV:
        """Each head's read of the state, ``[batch, seq, heads, value_dim]``, before the gated norm and ``out_proj``."""
        return value

    @EProperty(kernel("output"), select=1, description="The recurrent state leaving the layer", unavailable=needs_torch_kernels)
    def state_output(self, value: torch.Tensor) -> State:
        """The state after this call's last token, ``[batch, heads, key_dim, value_dim]``: what the next decode step starts from."""
        return value
