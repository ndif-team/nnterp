"""`TokenEProperty`: a value the model holds flat over tokens, served per invoke as ``[batch, seq, ...]``.

A mixture of experts routes ``[batch * seq, ...]`` tensors, batch-major: the
router's logits, the weights and indices the experts receive, the experts'
output. nnsight narrows a served tensor to an invoke's rows only when its
leading dim is the whole batch, which a flat token axis is only on a one-token
step; otherwise every invoke is served the whole run's tokens. A
`TokenEProperty` serves such a tensor as this invoke's ``[batch, seq, ...]``
rows, a view, so in-place edits land on the model's tensor and reach that
invoke alone, and splices an assignment back into the whole flat tensor. A
tensor already at the value's layout rank (DBRX's experts, which take and
return ``[batch, seq, hidden]``) passes through both ways.
"""

from __future__ import annotations

from typing import Any

import torch
from nnsight.intervention.envoy import Envoy
from nnsight.intervention.interleaver import Mediator

from .eproperty import EProperty


def _rows_of_invoke() -> tuple[int | None, list | None]:
    """The run's batch size and this invoke's ``[start, size]`` rows (``None`` when it has the whole batch)."""
    mediator = Mediator.current("rows")
    batcher = mediator.interleaver.batcher if mediator.interleaver is not None else None
    if batcher is None:
        return None, None
    return batcher.total, (mediator.batch_group if batcher.batching else None)


def rows(value: Any, rank: int) -> Any:
    """A tensor flat over tokens (one rank below ``rank``) as this invoke's ``[batch, seq, ...]``, a view; anything else as is."""
    if not isinstance(value, torch.Tensor) or value.dim() != rank - 1:
        return value
    total, group = _rows_of_invoke()
    if group is not None and value.shape[0] == group[1]:  # nnsight already narrowed it: one token per row
        return value.unflatten(0, (group[1], -1))
    whole = value.unflatten(0, (total, -1)) if total else value.unsqueeze(0)
    return whole if group is None else whole.narrow(0, group[0], group[1])


def splice(whole: Any, value: Any, rank: int) -> Any:
    """``value``, this invoke's ``[batch, seq, ...]``, written into ``whole`` where the model holds it flat over tokens."""
    if not isinstance(whole, torch.Tensor) or whole.dim() != rank - 1:
        return value
    flat = value.flatten(0, 1)
    if flat.shape[0] == whole.shape[0]:  # the whole batch, or rows nnsight narrowed and widens itself
        return flat
    total, group = _rows_of_invoke()
    seq = whole.shape[0] // total
    start, size = group[0] * seq, group[1] * seq
    return torch.cat([whole[:start], flat.to(whole.dtype), whole[start + size:]])


class TokenEProperty(EProperty):
    """An `EProperty` whose tensor the model may hold flat over tokens, ``[batch * seq, ...]``, one rank below its layout.

    The same descriptor in every other respect (path, ``select``,
    ``unavailable``, layout, ``support``, the repr line). The preprocess,
    postprocess and transform see the model's own tensor; the user sees and
    writes this invoke's ``[batch, seq, ...]``:

    * ``__get__`` reads as `EProperty` does, then serves `rows` of the result:
      a view, so an in-place edit lands on the model's tensor.
    * ``__set__`` splices the assigned ``[batch, seq, ...]`` into the whole
      tensor the location holds now (`splice`), then writes as `EProperty`
      does.

    Under two or more invokes the spliced tensor's leading dim is not the
    batch; nnsight keeps such an edit from PR #738 on (`Batcher._widen_tensor`).
    """

    def _rank(self) -> int:
        return len(self.dims)

    def __get__(self, obj: Envoy | None, owner: Any = None) -> Any:
        value = super().__get__(obj, owner)
        return value if obj is None else rows(value, self._rank())

    def __set__(self, obj: Envoy, value: Any) -> None:
        self._check(obj)
        key = self.path(obj)
        whole = self._pick(self.attribute(key), Mediator.value(self._resolve(obj, key)), self._selection(obj))
        super().__set__(obj, splice(whole, value, self._rank()))
