"""`Standard`, the envoy every component derives from, the two helpers for tuple outputs, and the walk `support` makes over a block list."""

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
    #: The `StandardizedTransformer` this envoy belongs to, set when the model is built: what a value
    #: reads the model's config, processor or family through (the vision tower's image values).
    _root: Any = None

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


def standard_children(block: Envoy) -> dict[str, Standard]:
    """The block's children that carry standard values, by standard name (the alias where one is bound)."""
    bound = {alias: block.__dict__[alias] for alias in block._aliases}  # what each alias is bound to, however deep
    names = {id(child): alias for alias, child in bound.items()}
    found = {names.get(id(child), name): child for name, child in block._named_children() if isinstance(child, Standard)}
    found.update((alias, child) for alias, child in bound.items() if isinstance(child, Standard) and alias not in found)
    return found


def block_hosts(blocks: Any) -> dict[str, list[str]]:
    """Standard-value hosts across every block: module name -> value names, in first-seen order.

    The union over the blocks, so a hybrid lists both ``self_attn`` and
    ``linear_attn`` and a block lacking one reports it as missing; a
    module no block has (OPT's ``mlp``) is not listed.
    """
    hosts: dict[str, dict[str, None]] = {}
    for block in blocks:
        for module, child in standard_children(block).items():
            hosts.setdefault(module, {}).update(dict.fromkeys(child.values()))
    return {module: list(names) for module, names in hosts.items()}


def block_support(block: Any, hosts: dict[str, list[str]]) -> dict[str, str | None]:
    """One block's values by dotted name, over ``hosts``: ``None`` when available, else the reason."""
    support: dict[str, str | None] = dict(block.support())
    present = standard_children(block)
    for module, names in hosts.items():
        envoy = present.get(module)
        reasons = envoy.support() if envoy is not None else {}
        for name in names:
            if envoy is None:
                support[f"{module}.{name}"] = f"no {module} module on this block"
            else:
                support[f"{module}.{name}"] = reasons.get(name, f"no {name} value on this block's {module}")
    return support


def blocks_support(blocks: Any, layer: int | None = None) -> dict[str, Any]:
    """The block values of a block list, the way ``support`` reports them.

    With ``layer``, that block's values by dotted name (``"layer_output"``,
    ``"self_attn.attention_probabilities"``). Without, each value -> ``None``
    when available on every block, else ``{layer: reason}`` for the blocks
    where it is not. Shared by the text model's blocks and a vision tower's.
    """
    hosts = block_hosts(blocks)
    if layer is not None:
        return block_support(blocks[layer], hosts)
    per_layer = [block_support(block, hosts) for block in blocks]
    support: dict[str, Any] = {}
    for name in per_layer[0]:
        missing = {i: reasons[name] for i, reasons in enumerate(per_layer) if reasons[name]}
        support[name] = missing or None
    return support
