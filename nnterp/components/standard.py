"""`Standard`, the envoy every component derives from, the axis names every layout is spelled with, the two helpers for tuple outputs, and the walk `support` makes over a block list."""

from __future__ import annotations

from typing import Any

import torch
from nnsight.intervention.envoy import Envoy

from .eproperty import EProperty

# -- axis names -----------------------------------------------------------------------
# Every layout is composed from these. An axis that is a fact about the model is named as
# the size attribute that holds it, so `axis_size` resolves it by name; the rest are not.

#: Sizes of the model, resolved by `axis_size`: the root's, or a block's own on `Attention`, `Moe`, `StateSpace`.
HIDDEN = "hidden_size"
VOCAB = "vocab_size"
HEADS = "num_heads"
KV_HEADS = "num_kv_heads"
HEAD_DIM = "head_dim"
QK_HEAD_DIM = "qk_head_dim"
INTERMEDIATE = "intermediate_size"
EXPERTS = "num_experts"
TOP_K = "top_k"
#: Not facts about the model: their size is the tensor's.
BATCH = "batch"
SEQ = "seq"
#: The batch and sequence axes flattened into one, as a block that reshapes around a sublayer holds the stream (OPT).
TOKENS = "tokens"
QUERY = "query"
KEY = "key"
#: A hyper-connection residual's parallel copies of the stream (DeepSeek-V4's ``hc_mult``); no envoy holds it.
STREAMS = "streams"
#: A recurrent mixer's heads and state sides, sized off the mixer's module; no envoy holds them.
MIXER_HEADS = "heads"
KEY_DIM = "key_dim"
VALUE_DIM = "value_dim"
STATE_DIM = "state_dim"
GROUPS = "groups"
CHANNELS = "channels"
#: A vision tower's images, patches and image tokens: not facts about the model.
IMAGES = "images"
PATCHES = "patches"
IMAGE_TOKENS = "image_tokens"
#: The tower's width, a size of the model resolved on `Vision`; a different quantity from `HIDDEN`, so named apart.
VISION_HIDDEN = "vision_hidden_size"


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


def absent(envoy: Envoy, name: str, where: str = "block") -> str | None:
    """Why a value at the child module ``name`` is unavailable: the envoy has no such child or alias. ``None`` when it has."""
    if name in envoy._aliases or any(child == name for child, _ in envoy._named_children()):
        return None
    return f"no {name} module on this {where}"


def unsized(envoy: Envoy, name: str) -> NotImplementedError:
    """The error a size raises when the module spells it no way the base reads."""
    return NotImplementedError(
        f"{type(envoy._module).__name__} holds no {name} under a name {type(envoy).__name__} reads; "
        f"the family's {type(envoy).__name__} subclass overrides `{name}`"
    )


def axis_size(envoy: Envoy, axis: str) -> int:
    """The size of layout axis ``axis`` on a value ``envoy`` serves.

    An axis that is a fact about the model is named as the size that holds
    it (`HIDDEN` is ``"hidden_size"``, `HEADS` ``"num_heads"``), so its size
    is that attribute: on ``envoy`` when its class declares it (an
    attention's ``num_heads``, a mixture's ``num_experts``), else on the
    first envoy above it that does, up to the model's root (``hidden_size``,
    ``vocab_size``)::

        axis_size(model.layers[3], HIDDEN)          # model.hidden_size
        axis_size(model.layers[3].self_attn, HEADS)  # this block's heads

    Only an envoy's class counts (a property, the root's
    `StandardizedProperty`), never an attribute of the wrapped module, which
    can share a name and hold another number.

    Raises:
        ValueError: no envoy from ``envoy`` up to the root declares ``axis``:
            it is not a fact about the model (the batch, the sequence, the
            pattern's keys), and its size is the tensor's.
    """
    host = envoy
    while host is not None:
        if hasattr(type(host), axis):
            return getattr(host, axis)
        host = host.parent
    raise ValueError(
        f"axis {axis!r} of a value on {envoy.path or 'the model'} is not a fact about the model: "
        "no envoy from there up to the root holds a size by that name"
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
