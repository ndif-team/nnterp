"""The forward order of a standardized model's values, measured by one probe trace.

Within one invoke the values have to be read in the order the forward reaches
them. That order is not written down anywhere: it is measured. A probe runs one
`scan` (meta tensors, no real weights needed) with one invoke for the prompt and
one empty invoke per value. Each empty invoke reads its value, which parks it
until the forward reaches that value's location, and then appends the value's
name and served location to a shared list, so the list comes out in forward
order. Workers parked on the same location resume in the order they were
written, not in any order the forward gives them, so values served at one
location (the queries, keys and values at the attention call's arguments) share
a rank: either read order is legal among them.

Blocks are probed per block shape, not per model: one representative block for
each distinct (block class, standard child classes, available values), so a
hybrid's linear-attention and softmax-attention blocks are each measured.
`StandardizedTransformer.order` and `StandardizedTransformer.rank` read the result.
"""

from __future__ import annotations

from typing import Any

import nnsight
from nnsight.intervention.source import STATE

from .components import recurrent
from .components.eproperty import DerivedEProperty, EProperty
from .components.standard import block_hosts, block_support, standard_children, values

#: What the probe runs on: any short text; the order does not depend on it.
PROMPT = "The quick brown fox"


def block_shapes(model: Any) -> dict[tuple, list[int]]:
    """The blocks grouped by shape: (block class, standard child classes, available values) -> layer indices."""
    hosts = block_hosts(model.layers)
    shapes: dict[tuple, list[int]] = {}
    for i, block in enumerate(model.layers):
        children = tuple(sorted((name, type(child)) for name, child in standard_children(block).items()))
        available = frozenset(name for name, reason in block_support(block, hosts).items() if reason is None)
        shapes.setdefault((type(block), children, available), []).append(i)
    return shapes


def _location(host: Any, prop: EProperty) -> str | None:
    """Where ``prop`` is served on ``host``, read inside the trace after the value (any drill is built by then).

    ``None`` for a derived value, which is computed from several locations: it gets a rank of its own.
    """
    if isinstance(prop, DerivedEProperty):
        return None
    return prop._resolve(host, prop.path(host))


def _restore_forwards(model: Any) -> Any:
    """Undo, after the probe, the forward instrumentation it added; returns a function that does so.

    Reading a value inside a forward instruments that forward, and the
    instrumented copy keeps the module globals it was built with, so a
    `route_kernels` after it would never reach that model. The probe is
    invisible otherwise, so it leaves every forward it found plain, plain.
    """
    plain = [
        module for module in model._module.modules()
        if (state := module.__dict__.get(STATE)) is None or not state.sourced
    ]

    def restore() -> None:
        for module in plain:
            state = module.__dict__.get(STATE)
            if state is not None and state.sourced:
                state.body, state.sourced, state.compiled = state.original, False, None

    return restore


def probe(model: Any, layers: list[int], run: str = "scan") -> list[tuple[int | None, str, str | None]]:
    """One run (``model.scan`` or ``model.trace``) over every available root value and the block values of ``layers``.

    Returns ``(layer, name, location)`` in the order the forward reached them,
    ``layer`` ``None`` for a root value. Values a checkpoint does not have
    (`support` gives a reason) are not probed.
    """
    targets: list[tuple[int | None, str, Any, EProperty]] = []
    for name, prop in values(type(model)).items():
        if prop.reason(model) is None:
            targets.append((None, name, model, prop))
    hosts = block_hosts(model.layers)
    for i in layers:
        block = model.layers[i]
        for dotted, reason in block_support(block, hosts).items():
            if reason is not None:
                continue
            *path, attr = dotted.split(".")
            host = block
            for part in path:
                host = getattr(host, part)
            targets.append((i, dotted, host, values(type(host))[attr]))

    restore = _restore_forwards(model)
    try:
        with getattr(model, run)() as tracer:
            seen = nnsight.save([])
            with tracer.invoke(PROMPT):
                pass
            for layer, name, host, prop in targets:
                with tracer.invoke():
                    getattr(host, prop.name)  # parks until the forward reaches it; nothing to save
                    seen.append((layer, name, _location(host, prop)))
    finally:
        restore()
    return list(seen)


def compute(model: Any) -> dict[int, dict[str, int]]:
    """Every available value's rank, by side: ``-1`` the root's values before the blocks, each layer its block's, ``num_layers`` the root's after.

    A rank counts distinct served locations in forward order, so values at
    one location share it. The root's values are numbered once across both
    sides; each block's from 0. The probe is a `scan`; a forward that cannot
    run on meta tensors (one that branches on its data: Granite's
    ``torch.equal``, OPT's data-dependent shapes, a grouped-mm mixture of
    experts in float32) is probed with a `trace` instead, which loads the
    weights of a model not yet dispatched.
    """
    shapes = block_shapes(model)
    representatives = sorted(group[0] for group in shapes.values())
    try:
        seen = probe(model, representatives)
    except Exception:
        try:
            seen = probe(model, representatives, run="trace")
        except Exception as error:
            raise RuntimeError(f"measuring the forward order of {model.repo_id} failed: {error}") from error

    # One rank per distinct location, in the order the locations were first reached; a derived value is its own.
    reached: dict[Any, int] = {}
    ranked = []
    for layer, name, location in seen:
        key = location if location is not None else (layer, name)
        ranked.append((layer, name, reached.setdefault(key, len(reached))))
    first_block = min(rank for layer, _, rank in ranked if layer is not None)

    def dense(layer: int | None) -> dict[str, int]:
        """``layer``'s values (the root's for ``None``), their ranks renumbered from 0."""
        entries = [(name, rank) for at, name, rank in ranked if at == layer]
        steps = sorted({rank for _, rank in entries})
        return {name: steps.index(rank) for name, rank in entries}

    num_layers = len(model.layers)
    root = dense(None)
    table: dict[int, dict[str, int]] = {-1: {}, num_layers: {}}
    for layer, name, rank in ranked:
        if layer is None:
            table[-1 if rank < first_block else num_layers][name] = root[name]
    for group in shapes.values():
        block = dense(group[0])
        table.update((i, block) for i in group)
    return table


def cached(model: Any) -> dict[int, dict[str, int]]:
    """`compute`, once per model and kernel binding: `route_kernels` and `chunk_per_token` change what fires, so they invalidate it."""
    generation, table = model._order or (None, None)
    if generation != recurrent.generation:
        table = compute(model)
        model._order = (recurrent.generation, table)
    return table
