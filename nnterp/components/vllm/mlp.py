"""`Mlp`: a vLLM feed-forward module, whose contribution is its output."""

from __future__ import annotations

import torch
from nnsight.intervention.envoy import Envoy

from .. import mlp as standard
from ..layer import Residual
from ..mlp import Neurons
from ..standard import absent
from .flat import Flat, on_transformers_backend

#: Why a mixture of experts has no activation or neurons on vLLM.
ROUTED = "routed experts, inside vLLM's fused MoE kernel: there is no single activation or down projection"
#: Why a gated MLP has no activation of its own on vLLM.
GATED = (
    "a gated MLP on vLLM: act(gate) is never a module's output (gate_up_proj feeds act_fn, an act-and-multiply "
    "kernel, whose output is already act(gate) * up), so it is not served; mlp_neurons is the product"
)


def activation_reason(envoy: Envoy) -> str | None:
    """Why ``envoy`` has no activation to serve: a mixture, a gate, or no ``act_fn`` or ``down_proj``; else ``None``."""
    if absent(envoy, "experts", "mlp") is None:
        return ROUTED
    if absent(envoy, "gate_up_proj", "mlp") is None or absent(envoy, "gate_proj", "mlp") is None:
        return GATED
    return absent(envoy, "act_fn", "mlp") or absent(envoy, "down_proj", "mlp")


def neurons_reason(envoy: Envoy) -> str | None:
    """Why there is no single down projection on ``envoy``: a mixture, or no ``down_proj``; else ``None``."""
    return ROUTED if absent(envoy, "experts", "mlp") is None else absent(envoy, "down_proj", "mlp")


class Mlp(standard.Mlp):
    """A vLLM feed-forward module: its contribution is its output.

    Both `mlp_activation` and `mlp_neurons` are read as the down projection's
    input: vLLM shares one activation module between every block, so
    ``act_fn``'s own output cannot say which block called it. On a gated MLP
    ``act_fn`` computes ``act(gate) * up`` in one kernel, so `mlp_activation`
    is unavailable (`GATED`). A mixture of experts has neither (`ROUTED`).
    """

    @Flat("down_proj.input", batch=on_transformers_backend, unavailable=activation_reason, description="The activation's output, [1, tokens, intermediate_size]; a plain MLP's only")
    def mlp_activation(self, value: torch.Tensor) -> Neurons:
        """``act(up_proj(x))`` on a plain MLP, ``[1, tokens, intermediate_size]``: read as the down projection's input, which it is."""
        return value

    @Flat("down_proj.input", batch=on_transformers_backend, unavailable=neurons_reason, description="What the down projection receives, [1, tokens, intermediate_size]")
    def mlp_neurons(self, value: torch.Tensor) -> Neurons:
        """The neurons' values the down projection sums, ``[1, tokens, intermediate_size]``: ``act(gate) * up``, which is
        ``act_fn``'s output on vLLM's fused gate, or the activation itself on a plain MLP."""
        return value

    @Flat("output", description="What the MLP adds to the residual stream, [1, tokens, hidden_size]")
    def mlp_output(self, value: torch.Tensor) -> Residual:
        return value
