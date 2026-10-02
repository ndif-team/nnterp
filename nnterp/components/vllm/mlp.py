"""`Mlp`: a vLLM feed-forward module, whose contribution is its output."""

from __future__ import annotations

import torch

from .. import mlp as standard
from ..layer import Residual
from .flat import Flat


class Mlp(standard.Mlp):
    """A vLLM feed-forward module: its contribution is its output."""

    @Flat("output", description="What the MLP adds to the residual stream, [1, tokens, hidden]")
    def mlp_output(self, value: torch.Tensor) -> Residual:
        return value
