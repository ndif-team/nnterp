"""`Mlp`: a feed-forward module, whose contribution is ``mlp_output``."""

from __future__ import annotations

from typing import Any

import torch

from .eproperty import EProperty
from .layer import Residual
from .standard import Standard, first_tensor, in_width, module_int, rewrap, unsized


class Mlp(Standard):
    """A feed-forward module. Its contribution is ``mlp_output``; its width, `intermediate_size`, is read off the module."""

    @property
    def intermediate_size(self) -> int:
        """Width of this MLP's hidden layer; on a mixture of experts, one routed expert's.

        Read off the module: its experts' ``intermediate_dim`` / ``expert_dim`` /
        ``intermediate_size`` / ``ffn_hidden_size``, else its own
        ``intermediate_size`` / ``ffn_dim``, else the down projection's input
        width. A family whose module spells it another way overrides this.
        """
        module = self._module
        experts = getattr(module, "experts", None)
        size = module_int(experts, "intermediate_dim", "expert_dim", "intermediate_size", "ffn_hidden_size") if experts is not None else None
        if size is None:
            size = module_int(module, "intermediate_size", "ffn_dim")
        if size is None:
            size = in_width(module, "down_proj", "c_proj", "dense_4h_to_h", "fc2", "fc_out", "w2")
        if size is None:
            raise unsized(self, "intermediate_size")
        return size

    @EProperty(key="output", description="What the MLP adds to the residual stream")
    def mlp_output(self, value: Any) -> Residual:
        """The MLP sublayer's contribution to the residual stream.

        A tensor even when the module returns a tuple (a mixture of experts
        returns router scores beside it). On a family whose MLP adds the
        residual inside the module, the family's subclass reads the
        pre-residual value instead. In-place edits and assignment reach the
        model.
        """
        return first_tensor(value)

    @mlp_output.postprocess
    def mlp_output(self, value: torch.Tensor) -> Any:
        return rewrap(self, value)
