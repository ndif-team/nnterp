"""`Layer` and `FusedLayer`: a vLLM decoder block, called with the residual stream or with its two halves."""

from __future__ import annotations

from typing import Any

import torch

from .. import layer as standard
from ..eproperty import EProperty
from ..layer import Residual
from .flat import Flat, unbatched


def argument(inputs: tuple, index: int, name: str) -> Any:
    """One argument of a call served as ``(args, kwargs)``, however the caller passed it; ``None`` when absent."""
    args, kwargs = inputs
    return kwargs[name] if name in kwargs else args[index] if index < len(args) else None


def with_argument(inputs: tuple, index: int, name: str, value: Any) -> tuple:
    """``inputs`` with that argument replaced, where the caller passed it."""
    args, kwargs = inputs
    if name in kwargs:
        return args, {**kwargs, name: value}
    return (*args[:index], value, *args[index + 1:]), kwargs


class Layer(standard.Layer):
    """A vLLM block that is called with the residual stream and returns it.

    Two things vary between the blocks that work this way, and a family
    states them: `STREAM`, where the stream sits in the block's call (GPT-2's
    block takes it alone, most take the positions first), and the base's
    ``returns_tuple``, for a block that returns the stream with something
    beside it (Exaone4's and Cohere's return ``(hidden_states, residual)``
    with the whole stream first, where a `FusedLayer` returns its two halves).
    """

    #: The index of the residual stream in the block's call.
    STREAM = 0

    def skip_with(self, hidden: torch.Tensor) -> None:
        """Skip this block, handing ``hidden`` (``[1, tokens, hidden]``) on as its residual stream.

        Safe on a request that runs alone. The engine batches whatever is in
        flight into one step, and a skipped block has to answer for all of
        it; see nnsight's vLLM guide before skipping on a shared engine.
        """
        hidden = hidden.squeeze(0)
        self.skip((hidden, hidden) if self.returns_tuple else hidden)

    @Flat("inputs", select=lambda envoy: envoy.STREAM, description="The residual stream entering the block, [1, tokens, hidden]")
    def layer_input(self, value: torch.Tensor) -> Residual:
        return value

    @Flat("output", select=lambda envoy: 0 if envoy.returns_tuple else None, description="The residual stream leaving the block, [1, tokens, hidden]")
    def layer_output(self, value: torch.Tensor) -> Residual:
        return value


class FusedLayer(standard.Layer):
    """A vLLM block with the residual add fused into the next norm (Llama, Qwen, Gemma, most families).

    It is called ``forward(positions, hidden_states, residual)`` and returns
    ``(hidden_states, residual)``: ``hidden_states`` is what the last sublayer
    produced and ``residual`` the stream before it, so the stream is their
    sum, entering and leaving. The first block is called with ``residual``
    ``None`` and the embeddings as ``hidden_states``.

    The sum is computed for the read, so it is nobody's buffer. An edit goes
    back as a change to ``hidden_states`` (the next norm adds the two), and
    a read that edits nothing changes nothing: the difference it hands back is
    exactly zero.
    """

    #: Where the two halves sit in the block's call: ``(index, name)``.
    HIDDEN = (1, "hidden_states")
    RESIDUAL = (2, "residual")

    def skip_with(self, hidden: torch.Tensor) -> None:
        """Skip this block, handing ``hidden`` (``[1, tokens, hidden]``) on as its residual stream.

        Safe on a request that runs alone. The engine batches whatever is in
        flight into one step, and a skipped block has to answer for all of
        it; see nnsight's vLLM guide before skipping on a shared engine.
        """
        hidden = hidden.squeeze(0)
        self.skip((hidden, torch.zeros_like(hidden)))

    @staticmethod
    def _stream(hidden: torch.Tensor, residual: torch.Tensor | None) -> torch.Tensor:
        return (hidden.clone() if residual is None else hidden + residual).unsqueeze(0)

    @staticmethod
    def _edited(hidden: torch.Tensor, residual: torch.Tensor | None, view: torch.Tensor, name: str) -> torch.Tensor:
        """``hidden_states`` carrying the difference between ``view`` and the stream as the model has it."""
        stream = unbatched(view, hidden, name)
        if residual is None:
            return stream
        return hidden + (stream - (hidden + residual))

    # -- the stream entering the block ---------------------------------------------

    @EProperty("inputs", description="The residual stream entering the block: hidden_states + residual, [1, tokens, hidden]")
    def layer_input(self, value: tuple) -> Residual:
        return self._stream(argument(value, *self.HIDDEN), argument(value, *self.RESIDUAL))

    def _with_input(self, inputs: tuple, view: torch.Tensor) -> tuple:
        hidden, residual = argument(inputs, *self.HIDDEN), argument(inputs, *self.RESIDUAL)
        return with_argument(inputs, *self.HIDDEN, self._edited(hidden, residual, view, f"{self.path}.layer_input"))

    @layer_input.postprocess
    def layer_input(self, value: torch.Tensor) -> tuple:
        return self._with_input(self.inputs, value)

    @layer_input.transform
    def layer_input(self, view: torch.Tensor, raw: tuple) -> tuple:
        return self._with_input(raw, view)

    # -- the stream leaving it -------------------------------------------------------

    @EProperty("output", description="The residual stream leaving the block: hidden_states + residual, [1, tokens, hidden]")
    def layer_output(self, value: tuple) -> Residual:
        return self._stream(*value)

    def _with_output(self, output: tuple, view: torch.Tensor) -> tuple:
        hidden, residual = output
        return self._edited(hidden, residual, view, f"{self.path}.layer_output"), residual

    @layer_output.postprocess
    def layer_output(self, value: torch.Tensor) -> tuple:
        return self._with_output(self.output, value)

    @layer_output.transform
    def layer_output(self, view: torch.Tensor, raw: tuple) -> tuple:
        return self._with_output(raw, view)
