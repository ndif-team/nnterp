"""`Moe`: a mixture of experts, an `Mlp` whose routing and experts are standard values.

Nearly every mixture in transformers computes the same thing::

    logits, w, idx = router(x)          # logits [tokens, experts]; w, idx [tokens, top_k]; tokens = batch * seq
    routed = experts(x, idx, w)         # [tokens, hidden]
    out = routed (+ shared(x))

and the experts module's call receives ``(hidden, top_k_index, top_k_weights)``.
So the values are read where the model consumes them: the logits at the
operation inside the router's forward that produces them (before any
scoring), the weights and indices as the experts' arguments, the per-slot
outputs inside transformers' shared experts implementation, the routed sum as
the experts' output. Every write lands on the tensor the model uses.

The routing tensors are flat over tokens; each value is served as this
invoke's ``[batch, seq, ...]`` (a `TokenEProperty`), a view,
so in-place edits land, and an assignment is spliced back into the flat
tensor.
"""

from __future__ import annotations

from typing import Any

import torch
from jaxtyping import Float, Int
from nnsight.intervention.envoy import Envoy
from torch import Tensor

from .tokens import TokenEProperty
from .layer import Residual
from .mlp import Mlp
from .standard import module_int, unsized

#: The router's logits, one column per expert (ZAYA adds a "skip" column).
RouterLogits = Float[Tensor, "batch seq experts"]
#: The weight each selected expert's output is scaled by, one per routing slot.
ExpertWeights = Float[Tensor, "batch seq top_k"]
#: The expert each routing slot sends the token to.
ExpertIndices = Int[Tensor, "batch seq top_k"]
#: Each routing slot's weighted expert output: slot ``j`` is ``weight[j] * expert[indices[j]](x)``.
ExpertOutputs = Float[Tensor, "batch seq top_k hidden"]

#: The operation inside a router's forward that produces the logits: ``router_logits = F.linear(x, self.weight)``.
LOGITS = "F_linear_0"
#: The call inside transformers' ``@use_experts_implementation`` wrapper: ``return experts_forward(self, ...)``.
DISPATCH = "experts_forward_1"
#: ``weighted_out.view(tokens, top_k, hidden)`` in transformers' ``grouped_mm`` and ``batched_mm`` experts forwards.
PER_SLOT = "weighted_out_view_0"

#: The child names a shared expert goes by; a family aliases its own to ``shared_experts``.
SHARED_NAMES = ("shared_experts", "shared_expert", "shared_mlp")


def mixture_reason(envoy: Envoy) -> str | None:
    """Why this module runs no mixture of experts, or ``None``: the one place a family decides it (`Moe.no_mixture`)."""
    return envoy.no_mixture()


def needs_grouped_experts(envoy: Envoy) -> str | None:
    """Why the per-slot expert outputs are unavailable, or ``None``: they exist only in the ``grouped_mm`` / ``batched_mm`` forwards."""
    reason = envoy.no_mixture()
    if reason:
        return reason
    implementation = envoy.experts._module.config._experts_implementation
    if implementation not in ("grouped_mm", "batched_mm"):
        return (
            f"read inside transformers' grouped_mm / batched_mm experts forward, but this model runs {implementation!r}; "
            "load with experts_implementation='grouped_mm' (the default) or 'batched_mm'"
        )
    return None


def no_shared_expert(envoy: Envoy) -> str | None:
    """Why ``shared_expert_output`` is unavailable on this mixture, or ``None``: it has no shared expert."""
    reason = envoy.no_mixture()
    if reason:
        return reason
    children = envoy._module._modules
    if any(children.get(name) is not None for name in SHARED_NAMES):
        return None
    return "this mixture has no shared expert"


class Moe(Mlp):
    """A mixture of experts: its routing, each routed expert's output, and the shared expert's.

    ``mlp_output`` is what the block adds, as on any `Mlp`. The rest, read
    where the model consumes them (``router``, ``experts`` and
    ``shared_experts`` are the standard child names; a family aliases its
    own):

    * ``router_logits`` ``[batch, seq, experts]``: before the scoring, at the
      operation in the router's forward that produces them (`LOGITS`), so a
      write changes the routing.
    * ``expert_weights`` / ``expert_indices`` ``[batch, seq, top_k]``: the
      experts module's arguments. Zero a weight to ablate that slot; write an
      index to reroute (the weight is not recomputed).
    * ``expert_outputs`` ``[batch, seq, top_k, hidden]``: each slot's weighted
      output, inside transformers' ``grouped_mm`` / ``batched_mm`` experts
      forward; unavailable under ``experts_implementation="eager"``.
    * ``routed_output``: the experts' sum, ``expert_outputs.sum(2)``.
    * ``shared_expert_output``: the shared expert's contribution, where there
      is one; ``routed_output + shared_expert_output`` is the mixture's output.

    Its sizes, `num_experts` and `top_k`, are read off the router or the
    experts module. `SCORING` says what the logits mean. A family whose module
    runs the mixture only on some checkpoints (Gemma-4's dense MLP, which hosts
    the block's experts) says why not in `no_mixture`, the one place every
    value's availability starts from.
    """

    #: How the router turns logits into weights: ``"softmax"`` (over every expert, then top-k),
    #: ``"topk_softmax"`` (top-k logits, then a softmax over them), ``"sigmoid"`` (per expert),
    #: ``"sparsemixer"`` (Phi-3.5-MoE), ``"hash"`` (the token id picks the experts: DeepSeek-V4's first blocks).
    SCORING = "softmax"

    def no_mixture(self) -> str | None:
        """Why this module runs no mixture of experts, or ``None`` (every `Moe` module does, unless its family says otherwise)."""
        return None

    # -- sizes (off the module) ---------------------------------------------------

    def _routing_modules(self) -> list[Any]:
        modules = self._module._modules
        found = [modules.get(name) for name in ("gate", "router", "experts")]
        for name in ("router", "experts"):  # handed down by the block where the mixture has no module of its own
            envoy = self.__dict__.get(name)
            if envoy is not None:
                found.append(envoy._module)
        return [self._module, *(module for module in found if module is not None)]

    @property
    def num_experts(self) -> int:
        """Routed experts in this mixture: the router's or experts module's ``num_experts`` / ``num_local_experts`` / ``n_routed_experts``."""
        for module in reversed(self._routing_modules()):
            size = module_int(module, "num_experts", "num_local_experts", "n_routed_experts")
            if size is not None:
                return size
        raise unsized(self, "num_experts")

    @property
    def top_k(self) -> int:
        """Experts each token is routed to: the module's or router's ``top_k`` / ``num_experts_per_tok``."""
        for module in self._routing_modules():
            size = module_int(module, "top_k", "num_experts_per_tok", "moe_topk")
            if size is not None:
                return size
        raise unsized(self, "top_k")

    # -- the routing --------------------------------------------------------------

    @TokenEProperty(f"router.source.{LOGITS}.output", description="The router's logits, one per expert, before the scoring", unavailable=mixture_reason)
    def router_logits(self, value: torch.Tensor) -> RouterLogits:
        """The router's logits, ``[batch, seq, experts]``, where the router's forward produces them.

        Before the scoring (`SCORING`) and any selection bias, so
        ``softmax(router_logits)`` is the routing distribution on a softmax
        router. Assign or edit in place: the router scores what is written,
        so the weights and indices follow.
        """
        return value

    @TokenEProperty("experts.inputs", select=2, description="The weight each selected expert's output is scaled by", unavailable=mixture_reason)
    def expert_weights(self, value: torch.Tensor) -> ExpertWeights:
        """The weights the experts receive, ``[batch, seq, top_k]``, one per routing slot.

        After the router's normalization and scaling: what multiplies each
        slot's expert output. Zero ``[b, t, j]`` to remove slot ``j`` of token
        ``t``; ``w.masked_fill(expert_indices == e, 0)`` removes expert ``e``.
        """
        return value

    @TokenEProperty("experts.inputs", select=1, description="The expert each routing slot sends the token to", unavailable=mixture_reason)
    def expert_indices(self, value: torch.Tensor) -> ExpertIndices:
        """The experts each token is routed to, ``[batch, seq, top_k]``, int64.

        Write an index to reroute that slot to another expert; its weight
        stays what the router gave the expert it chose.
        """
        return value

    @TokenEProperty(
        f"experts.source.{DISPATCH}.source.{PER_SLOT}.output",
        description="Each routing slot's weighted expert output", unavailable=needs_grouped_experts,
    )
    def expert_outputs(self, value: torch.Tensor) -> ExpertOutputs:
        """Each slot's expert output times its weight, ``[batch, seq, top_k, hidden]``, before the slots are summed.

        ``expert_outputs.sum(2) == routed_output``. The model's own tensor in
        transformers' ``grouped_mm`` (the default) and ``batched_mm`` experts
        forwards; unavailable under ``eager`` and hub kernels.
        """
        return value

    @TokenEProperty("experts.output", description="The routed experts' combined output", unavailable=mixture_reason)
    def routed_output(self, value: torch.Tensor) -> Residual:
        """The routed experts' weighted sum, ``[batch, seq, hidden]``, without the shared expert."""
        return value

    @TokenEProperty("shared_experts.output", description="The shared expert's output", unavailable=no_shared_expert)
    def shared_expert_output(self, value: torch.Tensor) -> Residual:
        """What the shared expert adds beside the routed ones, ``[batch, seq, hidden]``; unavailable on a mixture without one."""
        return value
