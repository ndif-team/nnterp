"""The descriptors a family's values are made of.

An `EProperty` is nnsight's ``eproperty`` with two additions: availability
(`Unavailable`, ``unavailable=``) and a *path* for a key, so one descriptor
serves a value wherever it lives: on the host module, on another module named
relative to it, or at an operation inside a forward. `DerivedEProperty`
computes one from several served values."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

from nnsight.intervention.envoy import Envoy
from nnsight.intervention.eproperty import WriteBack, eproperty
from nnsight.intervention.interleaver import Mediator
from nnsight.intervention.source import SourceNotAvailable
from nnsight.intervention.util import first_input, replace_first_input


class Unavailable(RuntimeError):
    """A standard value this checkpoint does not have, and why.

    Raised on access, before anything runs, so a wrong assumption fails at the
    line that makes it. `Standard.support` reports the same reasons without
    raising.
    """
    # TODO: make an unavailable value also answer False to ``hasattr``, with the
    # reason intact. Today ``hasattr`` *raises* this, since only AttributeError
    # counts as absence, and nnsight's ``Envoy.__getattr__`` rewrites an
    # AttributeError without the reason; an override of ``__getattr__`` on
    # `Standard` could re-raise `Unavailable` as an AttributeError carrying it.


class EProperty(eproperty):
    """An ``eproperty`` whose key is a path from the host, and which can say when it is unavailable.

    Args:
        key: Where the value lives, relative to the host envoy: dotted segments
            ending in ``output``, ``input`` or ``inputs``, or in another value
            the module at that point serves (an engine's ``logits``).
            ``"output"`` is the host's own output. A leading ``"../"`` (repeatable) steps to the
            parent module by native name, another name to a child module
            (aliases included) or, under a ``source`` segment, to an
            operation: ``"source"`` drills into the current module's or
            operation's forward, instrumenting it for this run, so
            ``"source.attention_interface_1.inputs"`` is the shared attention
            call's arguments. Above the host the location is the path's
            string, so ``"../source.hidden_states_view_0.output"`` is an
            operation of the parent block's own forward, which the family
            instruments at build (`Standard.sourced`); a call inside that
            forward cannot be drilled from a child. A function of the host returning such a path, for a
            forward that branches: it runs at read time, inside the trace, so it
            can read the forward's own branch variable (`RecurrentMixer.KERNEL`) or the
            config (a family's ``by_alibi``). ``None`` means the attribute's
            name, for a bare marker (`unavailable`).
        description: Shown in the model's repr, like any eproperty's.
        unavailable: A reason string, or a function of the envoy returning a
            reason or ``None``. A reason makes every read and write raise
            `Unavailable` before the model runs, and shows up in
            `Standard.support`. It is checked on the *instance*, so a
            checkpoint's config can decide (``attn_implementation``, an alibi
            variant), and a per-layer difference in a hybrid model too.
        select: One element of the served value: with ``inputs`` an int is a
            positional argument and a str a keyword; with ``output`` an int
            indexes the returned tuple. A write repacks the element into the
            current value, so assigning one argument of a call replaces just
            that argument. ``input`` is the call's first argument, ``inputs``
            with the first element selected. A function of the host returning
            one of those (or ``None``, the whole value) selects per access, for
            a value whose position differs between the calls a forward branches
            to (`SelectiveScan`'s prompt and decode kernels, `StateSpace`'s
            two kernels).

    The location is served by nnsight the way any eproperty's is, whatever the
    path: a module's output, a sibling norm's, or an operation's arguments,
    with ``preprocess``, ``postprocess`` and ``transform`` all available. An
    operation inside a called function only exists once someone has drilled
    into that call in the *current* run (the interleaver resolves the callee
    from the live value and clears what it built at the start of every run),
    so a path through ``source`` is walked before every read or write. An
    operation that is not there raises `SourceNotAvailable` naming what is,
    rather than the `AttributeError` a descriptor would otherwise swallow into
    "no attribute".
    """

    def __init__(
        self,
        key: str | Callable[[Envoy], str] | None = None,
        description: str | None = None,
        unavailable: str | Callable[[Envoy], str | None] | None = None,
        select: int | str | Callable[[Envoy], int | str | None] | None = None,
    ) -> None:
        self.locate = key if callable(key) else None
        self.unavailable = unavailable
        self.select = select
        super().__init__(key=f"<{key.__name__}>" if callable(key) else key, description=description)

    def __set_name__(self, owner: type, name: str) -> None:
        # A bare marker (`unavailable("...")`) is never called on a stub, so it
        # learns its name from the class body instead.
        if self.name is None:
            self.name = name
        if self.key is None:
            self.key = name

    # -- availability -----------------------------------------------------------

    def reason(self, obj: Envoy) -> str | None:
        """Why this value is not available on ``obj``, or ``None`` when it is."""
        return self.unavailable(obj) if callable(self.unavailable) else self.unavailable

    def _check(self, obj: Envoy) -> None:
        try:
            reason = self.reason(obj)
        except AttributeError as error:
            # Never let this surface as an AttributeError: a descriptor's
            # AttributeError falls through to Envoy.__getattr__ and comes back
            # as "no attribute 'name'", hiding the predicate's own bug.
            raise RuntimeError(
                f"the availability check of {obj.path}.{self.name} failed: {error}"
            ) from error
        if reason:
            raise Unavailable(f"{obj.path}.{self.name} is not available: {reason}")

    # -- the layout -------------------------------------------------------------

    @property
    def layout(self) -> Any:
        """The value's shape as a ``jaxtyping`` type (``Residual``, ``Pattern``, ...), or ``None``.

        Read off the return annotation of the function that defines the
        value. ``isinstance(tensor, value.layout)`` checks rank and dtype;
        `dims` names the axes, the same on every family.
        """
        import types
        import typing

        hint = self._hint()
        if isinstance(hint, types.UnionType):  # ``State | None``
            hint = next((arg for arg in typing.get_args(hint) if arg is not type(None)), None)
        return hint if hasattr(hint, "dim_str") else None

    def _hint(self) -> Any:
        """The resolved return annotation of the function that defines the value, or ``None``."""
        import typing

        func = self._preprocess
        return typing.get_type_hints(func, include_extras=True).get("return") if func is not None else None

    def _optional(self) -> bool:
        """Whether the annotation is ``Layout | None``."""
        import types

        return isinstance(self._hint(), types.UnionType)

    @property
    def dims(self) -> tuple[str, ...] | None:
        """The axis names of `layout`: ``("batch", "seq", "hidden")``."""
        layout = self.layout
        return tuple(layout.dim_str.split()) if layout is not None else None

    def __str__(self) -> str:
        """The repr line: ``(name) -> Layout [axes]: description`` for a value with a layout, else nnsight's line."""
        layout = self.layout
        if layout is None:
            return f"({self.name}): {self.description}"
        optional = " | None" if self._optional() else ""
        return f"({self.name}) -> {layout_name(layout)}{optional} [{' '.join(self.dims)}]: {self.description}"

    # -- the location -----------------------------------------------------------

    def path(self, obj: Envoy) -> str:
        """The key for ``obj``: the path itself, or what the key function returns for it."""
        return self.locate(obj) if self.locate is not None else self.key

    def inside_forward(self) -> bool:
        """Whether the value is an operation inside a forward (a key function, or a ``source`` segment on its path)."""
        return self.locate is not None or "source" in (self.key or "").lstrip("./").split(".")

    def _resolve(self, obj: Envoy, key: str) -> str:
        """The served location ``key`` names from ``obj``, walking (and drilling) the path."""
        up = 0
        while key.startswith("../"):
            up, key = up + 1, key[3:]
        *walk, attribute = key.split(".")
        if attribute == "inputs":  # the same location as ``input``, served whole
            attribute = "input"
        if up:
            # Above the host the path is arithmetic on names: an envoy knows its
            # own path but not its parent, and the parent's forward, when the
            # path goes into it, is instrumented already (`Standard.sourced`),
            # so the location is served by its string. One level of ``source``
            # is what that gives; a call inside the parent's forward would need
            # the parent drilled, which only an envoy can do.
            parts = obj.path.split(".")[:-up]
            if walk.count("source") > 1:
                raise ValueError(
                    f"{obj.path}.{self.name}: {key!r} drills into a call inside the parent's forward; "
                    "a path above the host reaches the parent's own operations only"
                )
            return ".".join([*parts, *walk, attribute])
        try:
            # Every segment is an attribute: a child module, ``source``, an operation.
            node = obj.get(".".join(walk)) if walk else obj
        except AttributeError as error:
            raise SourceNotAvailable(
                f"{obj.path}.{self.name} reads {key!r}, which this run does not have: "
                f"{error}. The forward took a path this family's toolkit does not expect."
            ) from None
        return f"{node.path}.{attribute}"

    # -- select -----------------------------------------------------------------

    def _selection(self, obj: Envoy) -> int | str | None:
        """The element this access selects: `select` itself, or what it returns for ``obj``."""
        return self.select(obj) if callable(self.select) else self.select

    def _pick(self, attribute: str, value: Any, select: int | str | None) -> Any:
        if attribute == "input":
            return first_input(*value)
        if select is None:
            return value
        if attribute == "inputs":
            args, kwargs = value
            return kwargs[select] if isinstance(select, str) else args[select]
        return value[select]

    def _put(self, attribute: str, current: Any, element: Any, select: int | str | None) -> Any:
        if attribute == "input":
            return replace_first_input(*current, element)
        if select is None:
            return element
        if attribute == "inputs":
            args, kwargs = current
            if isinstance(select, str):
                return args, {**kwargs, select: element}
            args = list(args)
            args[select] = element
            return tuple(args), kwargs
        current = list(current)
        current[select] = element
        return tuple(current)

    # -- read and write -----------------------------------------------------------
    # The key is computed once per access: a key function may read a served
    # value pinned to the current step (a forward's branch variable), and a
    # second evaluation after the read would run with the pin relaxed.

    def __get__(self, obj: Envoy | None, owner: Any = None) -> Any:
        if obj is None:
            return self
        self._check(obj)
        key = self.path(obj)
        location = self._resolve(obj, key)
        if self._transform is not None:
            # A write-back bound by an earlier read of this value is still
            # waiting: the view it holds is this read's value too, so an edit to
            # either is the edit that goes back (nnsight's eproperty does the same).
            mediator = Mediator.current(location)
            occurrence = mediator.wanted(location)
            waiting = mediator.transform
            if waiting is not None and waiting.key == (self, location, occurrence):
                return waiting.view
        select = self._selection(obj)  # before the read: a select function may read an earlier value of the call
        raw = Mediator.value(location)
        value = self._pick(key.rsplit(".", 1)[-1], raw, select)
        if self._preprocess is not None:
            value = self._preprocess(obj, value)
        if self._transform is not None:
            # Bound now so the user's in-place edits on the returned view are in
            # it when the worker next moves on and flushes it (nnsight's
            # eproperty does the same); the raw served value rides along for a
            # write-back that has to rebuild a container around the view.
            mediator.transform = WriteBack(partial(self._transform, obj, value, raw), value, self, location, occurrence)
        return value

    def __set__(self, obj: Envoy, value: Any) -> None:
        self._check(obj)
        if self._postprocess is not None:
            value = self._postprocess(obj, value)
        key = self.path(obj)
        location = self._resolve(obj, key)
        attribute = key.rsplit(".", 1)[-1]
        select = self._selection(obj)
        if select is not None or attribute == "input":
            value = self._put(attribute, Mediator.value(location), value, select)
        Mediator.swap(location, value)


def layout_name(layout: Any) -> str:
    """The name a layout alias is exported under (``Residual``, ``Pattern``), else the type's own name."""
    from .. import components, standardized

    for module in (components, standardized):
        for name, value in vars(module).items():
            if value is layout and not name.startswith("_"):
                return name
    return getattr(layout, "__name__", repr(layout))


def unavailable(reason: str) -> EProperty:
    """A value a family does not have: assign it in the class body in place of the inherited one.

    ``attention_probabilities = unavailable("no softmax: the attention is linear")``
    keeps the name in the tree and in the repr, with the reason, and makes any
    access raise `Unavailable` with it.
    """
    return EProperty(description=f"Unavailable: {reason}", unavailable=reason)


class DerivedEProperty(EProperty):
    """An `EProperty` computed from other served values rather than read at one location.

    ``compute(envoy)`` runs at read time inside the trace and may read any
    number of values; the result is read-only (assign through whatever
    method the host offers). Listed in the repr like any value.
    """

    def __init__(self, compute: Callable[[Envoy], Any], description: str | None = None, unavailable: Any = None) -> None:
        super().__init__(key=f"<{compute.__name__}>", description=description, unavailable=unavailable)
        self.compute = compute
        self._preprocess = compute  # what `layout` reads the annotation from; never called as a preprocess

    def __get__(self, obj: Envoy | None, owner: Any = None) -> Any:
        if obj is None:
            return self
        self._check(obj)
        return self.compute(obj)  # not `_preprocess(obj, value)`: there is no served value

    def __set__(self, obj: Envoy, value: Any) -> None:
        raise AttributeError(f"{self.name} is derived and read-only")
