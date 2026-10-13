"""`Mlp`: a vLLM feed-forward module, whose contribution is its output."""

from __future__ import annotations

import torch

from .. import mlp as standard
from ..eproperty import unavailable
from ..layer import Residual
from .flat import Flat


class Mlp(standard.Mlp):
    """A vLLM feed-forward module: its contribution is its output."""

    #: The activation's and the down projection's flat rows are not served yet.
    mlp_activation = unavailable("not mapped on vLLM yet: the projection takes flat [tokens, width] rows")
    mlp_neurons = unavailable("not mapped on vLLM yet: the projection takes flat [tokens, width] rows")

    @Flat("output", description="What the MLP adds to the residual stream, [1, tokens, hidden_size]")
    def mlp_output(self, value: torch.Tensor) -> Residual:
        return value
