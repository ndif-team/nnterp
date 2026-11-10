"""`Layer`: a decoder block, whose residual stream is ``layer_output``."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import torch
from nnsight.intervention.envoy import Envoy

from jaxtyping import Float
from torch import Tensor

from .eproperty import EProperty, Unavailable
from .standard import BATCH, HIDDEN, SEQ, STREAMS, TOKENS, Standard, absent, axis_size, first_tensor, rewrap

if TYPE_CHECKING:
    from .attention import Attention
    from .linear_attention import LinearAttention
    from .mlp import Mlp

#: The residual stream and everything added to it: ``layer_output``, the contributions, ``token_embeddings``, a sublayer's input.
Residual = Float[Tensor, f"{BATCH} {SEQ} {HIDDEN}"]
#: The residual stream with its batch and sequence axes flattened, as a block that reshapes around a sublayer holds
#: it (OPT's ``layer_mid`` and ``mlp_norm_output``).
FlatResidual = Float[Tensor, f"{TOKENS} {HIDDEN}"]
#: A hyper-connection residual (DeepSeek-V4): ``streams`` parallel copies of the stream, the block's own
#: ``[batch, seq, streams, hidden_size]``. ``layer_output`` (and a block's input) on those families, in place of `Residual`.
Streams = Float[Tensor, f"{BATCH} {SEQ} {STREAMS} {HIDDEN}"]
#: One weight per stream and token: how much of a sublayer's output each stream receives.
StreamWeights = Float[Tensor, f"{BATCH} {SEQ} {STREAMS}"]
#: A per-token ``streams x streams`` matrix mixing the streams: entry ``[j, k]`` is what stream ``j`` gives stream ``k``.
StreamMixing = Float[Tensor, f"{BATCH} {SEQ} {STREAMS} {STREAMS}"]


#: Reasons a family gives for a boundary its block does not have.
PARALLEL = "a parallel block: the attention and the MLP read the same input, so there is no stream between them"
POST_NORM_ATTENTION = "a post-norm block: the attention reads the residual stream itself, layer_input"
POST_NORM_MLP = "a post-norm block: the MLP reads the residual stream itself, layer_mid"
STATE_SPACE = "a state-space block: one sublayer, the mixer, so there is no stream between sublayers and no MLP"
ONE_SUBLAYER = "a one-sublayer block: its norm feeds its one sublayer, so there is no stream between sublayers"


class Layer(Standard):
    """A decoder block. Its hidden states are ``layer_output``, whatever the block returns.

    Attributes:
        self_attn: The softmax attention, an `Attention`; absent on a hybrid's linear blocks.
        linear_attn: The recurrent mixer, a `LinearAttention` (gated DeltaNet) or a `StateSpace` (Mamba-2);
            hybrids and state-space models only.
        mlp: The feed-forward, an `Mlp`; absent on OPT.
        input_layernorm, post_attention_layernorm: The block's norms under their aliased names,
            where the family has them (their meaning varies; see the README).

    Between ``layer_input`` and ``layer_output`` the block has four more boundaries,
    each the input or output of one of its modules: ``attention_norm_output``,
    ``layer_mid``, ``mlp_norm_output`` and ``mlp_input``. They are defined on the
    pre-norm block (``x + attn(norm1(x))``, then ``+ mlp(norm2(.))``) and keyed on
    the aliased norms, so on a block built otherwise the family's subclass points
    them at its own modules or marks them unavailable with the reason; a block
    without the module reports ``no <module> module on this block``.
    """

    self_attn: Attention
    linear_attn: LinearAttention
    mlp: Mlp
    input_layernorm: Envoy
    post_attention_layernorm: Envoy

    #: Whether the block returns ``(hidden_states, ...)`` rather than the tensor alone.
    #: What `skip_with` has to hand back in the block's place; a family that
    #: returns a tuple says so.
    returns_tuple = False

    def skip_with(self, hidden: torch.Tensor) -> None:
        """Skip this block, handing ``hidden`` on as its residual stream.

        The block does not run; ``hidden`` takes the place of its
        ``layer_output``, packed the way the block would have returned it (a
        tuple family gets ``(hidden, None)``: the second element is the
        attention weights nothing downstream reads). Call it inside a trace,
        before the block runs.
        """
        self.skip((hidden, None) if self.returns_tuple else hidden)

    @property
    def attention_input_gain(self) -> Tensor:
        """The per-channel gain between the residual stream and what the attention reads, ``[hidden_size]``.

        The gain of the norm `attention_norm_output` is read at (``weight``;
        ``1 + weight`` on Gemma; ones on a norm without one), so the attention
        reads ``gain * n(x)``, plus the norm's bias where it has one, for a
        stream ``x`` and its normalization ``n`` (``x / rms(x)``, or ``x``
        centred first on a LayerNorm). On a post-norm block the attention reads
        the stream itself, and the gain is ones. Folding it into a head's
        weights, ``value_weight(kv) @ (gain * h)``, is that head's map of a
        stream direction ``h``.

        Measured by calling the norm, outside a trace or inside one, on
        ``+-256`` in alternating channels, which it normalizes to ``+-1``
        whatever it computes in: exact on a norm without a bias, to the
        rounding of the bias otherwise. In the dtype and on the device of the
        norm's weight (else the block's).

        Raises:
            Unavailable: with `attention_norm_output`'s reason, where the block
                has no stream entering an attention (a mixture of hyper-connection
                streams, an MLP-only block).
        """
        value = type(self).attention_norm_output
        reason = value.reason(self)
        if reason and reason != POST_NORM_ATTENTION:
            raise Unavailable(f"{self.path}.attention_input_gain is not available: {reason}")
        block = next(self._module.parameters())
        if reason:  # a post-norm block: the attention reads the stream itself
            return torch.ones(axis_size(self, HIDDEN), dtype=block.dtype, device=block.device)
        norm = self.get(value.path(self).removesuffix(".output"))
        weight = next(norm._module.parameters(), block)
        signs = torch.ones(axis_size(self, value.dims[-1]), dtype=weight.dtype, device=weight.device)
        signs[1::2] = -1
        return (norm(signs * 256) - norm(signs * -256)) / 2 * signs

    @EProperty(key="input", description="The residual stream entering the block")
    def layer_input(self, value: torch.Tensor) -> Residual:
        """The residual stream entering this block, before any of its norms.

        The block's first argument where the block is called with the stream
        (every transformers family), and the stream wherever an engine passes
        it otherwise (vLLM's fused blocks take it in two halves). With the
        contributions it is the identity the values are defined by:
        ``layer_input + attention_output + mlp_output == layer_output``.
        In-place edits and assignment reach the model.
        """
        return value

    @EProperty("input_layernorm.output", description="The normed stream entering the attention", unavailable=partial(absent, name="input_layernorm"))
    def attention_norm_output(self, value: torch.Tensor) -> Residual:
        """The attention's pre-norm output: what the attention is called with, ``[batch, seq, hidden_size]``.

        ``input_layernorm.output`` on a pre-norm block. In-place edits and
        assignment reach the model.
        """
        return value

    @EProperty("post_attention_layernorm.input", description="The residual stream after the attention add", unavailable=partial(absent, name="post_attention_layernorm"))
    def layer_mid(self, value: torch.Tensor) -> Residual:
        """The residual stream between the two sublayers, ``[batch, seq, hidden_size]``.

        What the MLP's pre-norm is called with: ``layer_input +
        attention_output`` on a pre-norm block, where it is read as
        ``post_attention_layernorm.input``. A parallel block has no such
        stream, and marks this unavailable. An in-place edit changes the stream
        itself; an assignment replaces the norm call's argument, and the block
        has already bound the stream to its residual (``residual =
        hidden_states``), so the MLP reads the assigned tensor while the
        residual add keeps the old one.
        """
        return value

    @EProperty("post_attention_layernorm.output", description="The normed stream entering the MLP", unavailable=partial(absent, name="post_attention_layernorm"))
    def mlp_norm_output(self, value: torch.Tensor) -> Residual:
        """The MLP's pre-norm output, ``[batch, seq, hidden_size]``: ``post_attention_layernorm.output`` on a pre-norm block.

        The same tensor as `mlp_input` where nothing sits between the norm
        and the MLP. In-place edits and assignment reach the model.
        """
        return value

    @EProperty("mlp.input", description="What the MLP is called with", unavailable=partial(absent, name="mlp"))
    def mlp_input(self, value: torch.Tensor) -> Residual:
        """The MLP's first argument, ``[batch, seq, hidden_size]``. In-place edits and assignment reach the model."""
        return value

    @EProperty(key="output", description="The residual stream leaving the block, a tensor even when the block returns a tuple")
    def layer_output(self, value: Any) -> Residual:
        """The residual stream leaving this block, always a tensor.

        Some blocks return ``hidden_states`` alone, others a tuple with it first
        (GPT-J, Bloom, MPT, Falcon). This is the tensor either way, the same
        object the block returned, so in-place edits reach the model. Assigning
        replaces it and, for a tuple block, leaves the other elements as they
        were::

            with model.trace(prompt):
                resid = model.layers[3].layer_output.save()
                model.layers[3].layer_output[:, -1] = 0
                model.layers[4].layer_output = resid * 2
        """
        return first_tensor(value)

    @layer_output.postprocess
    def layer_output(self, value: torch.Tensor) -> Any:
        return rewrap(self, value)
