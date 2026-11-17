"""`Moe`: a mixture of experts, an `Mlp` whose routing and experts are standard values."""

from __future__ import annotations

from typing import Any

import torch
from jaxtyping import Float, Int
from nnsight.intervention.envoy import Envoy
from torch import Tensor

from .tokens import TokenEProperty
from .layer import Residual
from .eproperty import EProperty
from .mlp import Mlp, Neurons
from .standard import BATCH, EXPERTS, HIDDEN, SEQ, TOP_K, absent, module_int, unsized

#: The router's logits, one column per expert (ZAYA adds a "skip" column).
RouterLogits = Float[Tensor, f"{BATCH} {SEQ} {EXPERTS}"]
#: The weight each selected expert's output is scaled by, one per routing slot.
ExpertWeights = Float[Tensor, f"{BATCH} {SEQ} {TOP_K}"]
#: The expert each routing slot sends the token to.
ExpertIndices = Int[Tensor, f"{BATCH} {SEQ} {TOP_K}"]
#: Each routing slot's weighted expert output: slot ``j`` is ``weight[j] * expert[indices[j]](x)``.
ExpertOutputs = Float[Tensor, f"{BATCH} {SEQ} {TOP_K} {HIDDEN}"]

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


#: Why a mixture's ``mlp_activation`` and ``mlp_neurons`` are unavailable.
ROUTED = "routed experts; read per-expert via the Moe component"


def routed(name: str):
    """Why a dense-MLP value at the child ``name`` is unavailable on a `Moe`: `ROUTED` where it runs a mixture, else `absent`."""
    def reason(envoy: Envoy) -> str | None:
        return absent(envoy, name, "mlp") if envoy.no_mixture() else ROUTED
    return reason


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

    The values are read where the model consumes them, on the shape nearly
    every mixture in transformers has::

        logits, w, idx = router(x)          # logits [tokens, num_experts]; w, idx [tokens, top_k]; tokens = batch * seq
        routed = experts(x, idx, w)         # [tokens, hidden_size]
        out = routed (+ shared(x))

    so every write lands on the tensor the model uses. ``router``,
    ``experts`` and ``shared_experts`` are the standard child names; a family
    aliases its own. `SCORING` says what the logits mean. A family whose
    module runs the mixture only on some checkpoints says why not in
    `no_mixture`, the one place every value's availability starts from.
    """

    # A routed MLP has no single activation or down projection; a module that runs no mixture on this
    # checkpoint (Gemma-4's dense MLP, GraniteMoE-Hybrid's shared expert alone) is a dense MLP.
    @EProperty("act_fn.output", description=Mlp.mlp_activation.description, unavailable=routed("act_fn"))
    def mlp_activation(self, value: torch.Tensor) -> Neurons:
        return value

    @EProperty("down_proj.input", description=Mlp.mlp_neurons.description, unavailable=routed("down_proj"))
    def mlp_neurons(self, value: torch.Tensor) -> Neurons:
        return value


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
        for name in ("router", "experts"):  # the block's, where the mixture has no module of its own (a family property)
            envoy = getattr(self, name, None)
            if isinstance(envoy, Envoy) and all(envoy._module is not module for module in found):
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
        """Experts each token is routed to: the module's or router's ``top_k`` / ``num_experts_per_tok`` / ``moe_topk``."""
        for module in self._routing_modules():
            size = module_int(module, "top_k", "num_experts_per_tok", "moe_topk")
            if size is not None:
                return size
        raise unsized(self, "top_k")

    # -- the routing --------------------------------------------------------------

    @TokenEProperty(f"router.source.{LOGITS}.output", description="The router's logits, one per expert, before the scoring", unavailable=mixture_reason)
    def router_logits(self, value: torch.Tensor) -> RouterLogits:
        """The router's logits, ``[batch, seq, num_experts]``, where the router's forward produces them.

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
        stays what the router gave the expert it chose. Pick an expert the
        token's other slots do not use: JetMoE's forward raises on a repeated
        one, and the other families run that expert once per slot.
        """
        return value

    @TokenEProperty(
        f"experts.source.{DISPATCH}.source.{PER_SLOT}.output",
        description="Each routing slot's weighted expert output", unavailable=needs_grouped_experts,
    )
    def expert_outputs(self, value: torch.Tensor) -> ExpertOutputs:
        """Each slot's expert output times its weight, ``[batch, seq, top_k, hidden_size]``, before the slots are summed.

        ``expert_outputs.sum(2) == routed_output``. The model's own tensor in
        transformers' ``grouped_mm`` (the default) and ``batched_mm`` experts
        forwards; unavailable under ``eager`` and hub kernels.
        """
        return value

    @TokenEProperty("experts.output", description="The routed experts' combined output", unavailable=mixture_reason)
    def routed_output(self, value: torch.Tensor) -> Residual:
        """The routed experts' weighted sum, ``[batch, seq, hidden_size]``, without the shared expert."""
        return value

    @TokenEProperty("shared_experts.output", description="The shared expert's output", unavailable=no_shared_expert)
    def shared_expert_output(self, value: torch.Tensor) -> Residual:
        """What the shared expert adds beside the routed ones, ``[batch, seq, hidden_size]``; unavailable on a mixture without one."""
        return value
