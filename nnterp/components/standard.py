"""`Standard`, the envoy every component derives from, and the two helpers for tuple outputs."""

from __future__ import annotations

from typing import Any

import torch
from nnsight.intervention.envoy import Envoy

from .eproperty import EProperty


def first_tensor(value: Any) -> torch.Tensor:
    """The tensor of a module output: the first element when it is a tuple."""
    return value[0] if isinstance(value, tuple) else value


def rewrap(envoy: Envoy, value: torch.Tensor) -> Any:
    """``value`` in the shape the module returned: back in its tuple, if any."""
    current = envoy.output
    return (value, *current[1:]) if isinstance(current, tuple) else value


def module_int(module: Any, *names: str) -> int | None:
    """The first of ``names`` the module holds as an integer, or ``None``."""
    for name in names:
        value = getattr(module, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def in_width(module: Any, *names: str) -> int | None:
    """The input width of the first of ``names`` the module has as a projection (``nn.Linear``, or transformers' ``Conv1D``), or ``None``."""
    for name in names:
        projection = getattr(module, name, None)
        width = getattr(projection, "in_features", None) or getattr(projection, "nx", None)
        if isinstance(width, int):
            return width
    return None


def unsized(envoy: Envoy, name: str) -> NotImplementedError:
    """The error a size raises when the module spells it no way the base reads."""
    return NotImplementedError(
        f"{type(envoy._module).__name__} holds no {name} under a name {type(envoy).__name__} reads; "
        f"the family's {type(envoy).__name__} subclass overrides `{name}`"
    )


def values(cls: type) -> dict[str, EProperty]:
    """A class's standard values by name, base classes first."""
    found: dict[str, EProperty] = {}
    for klass in reversed(cls.__mro__):
        for name, attr in vars(klass).items():
            if isinstance(attr, EProperty):
                found[name] = attr
    return found


class Standard(Envoy):
    """An envoy carrying standard values: what `Layer`, `Attention` and `Mlp` share.

    A value that is an operation inside a forward is served only on a call
    whose forward was instrumented before the call began. Instrumenting on
    first read is enough when the read comes before the module runs, which
    is the usual case; it is too late when the worker is already parked
    inside the module, as it is when a block's ``mlp_output`` (an operation
    in the block's forward) is read after the block's ``attention_output``.
    A family whose value sits in such a forward sets ``sourced = True`` on
    that envoy, and the forward is instrumented when the envoy is built and
    again when real weights replace meta ones, which reinstalls the plain
    forward.
    """

    #: Instrument this module's forward at build, ahead of any run (see the class docstring).
    sourced = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.sourced:
            self.source

    def _update(self, module: Any) -> None:
        super()._update(module)
        if self.sourced:
            self.source

    @classmethod
    def values(cls) -> dict[str, EProperty]:
        """This class's standard values by name, base classes first."""
        return values(cls)

    def support(self) -> dict[str, str | None]:
        """Each standard value here -> ``None`` when available, else the reason."""
        return {name: value.reason(self) for name, value in self.values().items()}
