"""`Flat`: a ``[tokens, ...]`` tensor of one vLLM request, served in nnterp's layout as a private copy."""

from __future__ import annotations

import functools
from typing import Any, Callable

import torch
from nnsight.intervention.envoy import Envoy
from nnsight.intervention.interleaver import Mediator

from ..eproperty import EProperty


def batched(rows: torch.Tensor, head_dim: int | None = None, heads_first: bool = False) -> torch.Tensor:
    """``[tokens, ...]`` as a private ``[1, tokens, ...]`` copy; with ``head_dim``, the last axis split into heads.

    ``[tokens, heads * head_dim]`` (or ``[tokens, heads, head_dim]``, where a
    module keeps its heads apart) becomes ``[1, tokens, heads, head_dim]``, or
    ``[1, heads, tokens, head_dim]`` with ``heads_first``.
    """
    view = rows.clone().unsqueeze(0)
    if head_dim is None:
        return view
    view = view.flatten(2).unflatten(-1, (-1, head_dim))
    return view.transpose(1, 2) if heads_first else view


def unbatched(value: torch.Tensor, rows: torch.Tensor, name: str, head_dim: int | None = None, heads_first: bool = False) -> torch.Tensor:
    """What `batched` served, back as the ``[tokens, ...]`` the model holds, refusing another shape.

    A written value is spliced into the step the engine is running, among
    other requests' rows. One of the wrong height would reach the next kernel
    as it is, where the mismatch can end the engine and every request in it,
    so it is refused here, in the block that wrote it.
    """
    expected = (1, *rows.shape)
    if head_dim is not None:
        tokens, heads = rows.shape[0], rows[0].numel() // head_dim
        expected = (1, heads, tokens, head_dim) if heads_first else (1, tokens, heads, head_dim)
    if value.shape != expected:
        raise ValueError(
            f"{name} is {expected} on this request and cannot be replaced by a value of shape "
            f"{tuple(value.shape)}; write into the rows you mean instead (``value[:, positions] = ...``)"
        )
    if head_dim is not None and heads_first:
        value = value.transpose(1, 2)
    return value.reshape(rows.shape).to(rows.dtype)


class Flat(EProperty):
    """An `EProperty` over a ``[tokens, ...]`` tensor, served as a private ``[1, tokens, ...]`` copy.

    Takes what `EProperty` takes (a path for a key, ``select`` for one
    argument of a call). The decorated function receives the batched copy and
    returns the value; its annotation is the layout. In-place edits are handed
    back to the model when the block moves on, and an assignment is checked
    against the rows this request has before it is swapped in.

    Args:
        heads: For a ``[tokens, heads * head_dim]`` tensor, where the head
            axis goes: ``"first"`` serves ``[1, heads, tokens, head_dim]``
            (the queries, keys and values), ``"last"`` ``[1, tokens, heads,
            head_dim]`` (the head outputs). The width of a head is the
            ``head_size`` of the module the key names, vLLM's attention layer.
        batch: Whether the tensor already carries the batch axis, ``[1,
            tokens, ...]``, as it does inside the transformers model vLLM's
            transformers backend runs: its rows are then ``tensor[0]``, served
            and handed back the same way. A function of the host for a value
            whose model decides (the root's ``token_embeddings``).
        factor: For a value that is the tensor times a scalar of the model
            (Granite's ``residual_multiplier``), a function of the host
            giving that scalar. The value is served multiplied and is a
            computed copy: a write goes back divided by the factor, and a
            read that edits nothing hands the model's tensor back untouched,
            so dividing never rounds a clean forward.
    """

    def __init__(
        self, *args: Any, heads: str | None = None, batch: bool | Callable[[Envoy], bool] = False,
        factor: Callable[[Envoy], float] | None = None, **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.heads = heads
        self.batch = batch
        self.factor = factor

    def _batched(self, obj: Envoy) -> bool:
        """Whether the tensor ``obj`` serves here carries the batch axis already."""
        return self.batch(obj) if callable(self.batch) else self.batch

    def _layout(self, obj: Envoy) -> dict:
        """How this value's rows are laid out for the reader: the keyword arguments of `batched` and `unbatched`."""
        if self.heads is None:
            return {}
        *walk, _ = self.path(obj).split(".")
        layer = obj.get(".".join(walk)) if walk else obj
        return {"head_dim": layer._module.head_size, "heads_first": self.heads == "first"}

    def __call__(self, preprocess: Callable) -> "Flat":
        @functools.wraps(preprocess)
        def read(envoy: Envoy, value: torch.Tensor) -> torch.Tensor:
            return preprocess(envoy, self._view(envoy, value))

        super().__call__(read)
        self._transform = self._hand_back
        return self

    def _view(self, obj: Envoy, value: torch.Tensor) -> torch.Tensor:
        """What a read of ``value`` serves: its rows as a private ``[1, tokens, ...]`` copy, times the factor."""
        view = batched(value[0] if self._batched(obj) else value, **self._layout(obj))
        return view if self.factor is None else view * self.factor(obj)

    def _rows(self, obj: Envoy, value: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        if self.factor is not None:
            value = value / self.factor(obj)
        if self._batched(obj):
            return unbatched(value, rows[0], f"{obj.path}.{self.name}", **self._layout(obj)).unsqueeze(0)
        return unbatched(value, rows, f"{obj.path}.{self.name}", **self._layout(obj))

    def _hand_back(self, obj: Envoy, view: torch.Tensor, raw: Any) -> Any:
        attribute, select = self.attribute(self.path(obj)), self._selection(obj)
        rows = self._pick(attribute, raw, select)
        if self.factor is not None and torch.equal(view, self._view(obj, rows)):
            return raw  # nothing edited: dividing the product back would round
        return self._put(attribute, raw, self._rows(obj, view, rows), select)

    def __set__(self, obj: Envoy, value: torch.Tensor) -> None:
        self._check(obj)
        key = self.path(obj)
        location, attribute, select = self._resolve(obj, key), self.attribute(key), self._selection(obj)
        current = Mediator.value(location)
        Mediator.swap(location, self._put(attribute, current, self._rows(obj, value, self._pick(attribute, current, select)), select))
