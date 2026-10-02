"""`Layer`: a decoder block, whose residual stream is ``layer_output``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from nnsight.intervention.envoy import Envoy

from jaxtyping import Float
from torch import Tensor

from .eproperty import EProperty
from .standard import Standard, first_tensor, rewrap

if TYPE_CHECKING:
    from .attention import Attention
    from .linear_attention import LinearAttention
    from .mlp import Mlp

#: The residual stream and everything added to it: ``layer_output``, the contributions, ``token_embeddings``, a sublayer's input.
Residual = Float[Tensor, "batch seq hidden"]
#: A hyper-connection residual (DeepSeek-V4): ``streams`` parallel copies of the stream, the block's own
#: ``[batch, seq, streams, hidden]``. ``layer_output`` (and a block's input) on those families, in place of `Residual`.
Streams = Float[Tensor, "batch seq streams hidden"]
#: One weight per stream and token: how much of a sublayer's output each stream receives.
StreamWeights = Float[Tensor, "batch seq streams"]
#: A per-token ``streams x streams`` matrix mixing the streams: entry ``[j, k]`` is what stream ``j`` gives stream ``k``.
StreamMixing = Float[Tensor, "batch seq streams streams"]


class Layer(Standard):
    """A decoder block. Its hidden states are ``layer_output``, whatever the block returns.

    Attributes:
        self_attn: The softmax attention, an `Attention`; absent on a hybrid's linear blocks.
        linear_attn: The recurrent mixer, a `LinearAttention` (gated DeltaNet) or a `StateSpace` (Mamba-2);
            hybrids and state-space models only.
        mlp: The feed-forward, an `Mlp`; absent on OPT.
        input_layernorm, post_attention_layernorm: The block's norms under their aliased names,
            where the family has them (their meaning varies; see the README).
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
