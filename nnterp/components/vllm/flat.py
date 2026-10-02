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
    """

    def __init__(self, *args: Any, heads: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.heads = heads

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
            return preprocess(envoy, batched(value, **self._layout(envoy)))

        super().__call__(read)
        self._transform = self._hand_back
        return self

    def _rows(self, obj: Envoy, value: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        return unbatched(value, rows, f"{obj.path}.{self.name}", **self._layout(obj))

    def _hand_back(self, obj: Envoy, view: torch.Tensor, raw: Any) -> Any:
        attribute, select = self.path(obj).rsplit(".", 1)[-1], self._selection(obj)
        return self._put(attribute, raw, self._rows(obj, view, self._pick(attribute, raw, select)), select)

    def __set__(self, obj: Envoy, value: torch.Tensor) -> None:
        self._check(obj)
        key = self.path(obj)
        location, attribute, select = self._resolve(obj, key), key.rsplit(".", 1)[-1], self._selection(obj)
        current = Mediator.value(location)
        Mediator.swap(location, self._put(attribute, current, self._rows(obj, value, self._pick(attribute, current, select)), select))
