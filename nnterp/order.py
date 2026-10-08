"""The forward order of a standardized model's values, measured by one probe run (see `probe`)."""

from __future__ import annotations

from typing import Any

import nnsight
from nnsight.intervention.source import STATE

from .components import recurrent
from .components.eproperty import DerivedEProperty
from .components.standard import standard_children, values


def available(host: Any, prefix: str = "") -> list[tuple[str, Any, Any]]:
    """``(name, host, value)`` for every value ``host`` has (`support` gives no reason)."""
    return [(prefix + name, host, prop) for name, prop in values(type(host)).items() if prop.reason(host) is None]


def block_values(block: Any) -> list[tuple[str, Any, Any]]:
    """`available` over a block and its standard children, by dotted name (``"self_attn.attention_queries"``)."""
    found = available(block)
    for name, child in standard_children(block).items():
        found += available(child, f"{name}.")
    return found


def probe(model: Any, targets: list[tuple[int | None, str, Any, Any]], run: Any) -> list[tuple[int | None, str, Any]]:
    """``(layer, name, location)`` for each target, in the order the forward reached them.

    One ``run`` (``model.scan``, meta tensors and no weights needed, or
    ``model.trace``) with one invoke for a prompt and one empty invoke per
    value. Each empty invoke reads its value, which parks it until the forward
    reaches that value's location, then appends the location it was served at,
    so the list comes out in forward order. Workers parked on one location
    resume in the order they were written, not in any order the forward gives
    them, so values served at one location (the attention call's queries, keys
    and values) tie: either read order is legal. A derived value, computed from
    several locations, is its own location ``(layer, name)``.
    """
    # Reading a value instruments its forward, and the instrumented copy keeps the module globals it
    # was built with, so a later `route_kernels` would never reach it: put back every forward found plain.
    # TODO: delete this once nnsight's `.source` stops snapshotting module globals (function_like copies fn.__globals__).
    plain = [m for m in model._module.modules() if not getattr(m.__dict__.get(STATE), "sourced", False)]
    try:
        with run() as tracer:
            seen = nnsight.save([])
            with tracer.invoke("The quick brown fox"):  # any short text: the order does not depend on it
                pass
            for layer, name, host, prop in targets:
                with tracer.invoke():
                    getattr(host, prop.name)  # parks until the forward reaches it; nothing to save
                    # Resolved after the read, when any drill is built.
                    location = (layer, name) if isinstance(prop, DerivedEProperty) else prop._resolve(host, prop.path(host))
                    seen.append((layer, name, location))
    finally:
        for m in plain:
            if (state := m.__dict__.get(STATE)) is not None and state.sourced:
                state.body, state.sourced, state.compiled = state.original, False, None
    return list(seen)


def compute(model: Any) -> dict[int, dict[str, int]]:
    """Every available value's rank, by side: ``-1`` the root's values before the blocks, each layer its block's, ``num_layers`` the root's after.

    One block of each shape (block class and the values it has) is probed, so a
    hybrid's linear- and softmax-attention blocks are each measured. A rank
    counts distinct locations in forward order: from 0 in each block, and once
    across the root's two sides. A forward that cannot run on meta tensors
    (one that branches on its data: Granite's ``torch.equal``, OPT's
    data-dependent shapes, a grouped-mm mixture of experts in float32) is
    probed with a `trace`, which loads the weights of a model not yet dispatched.
    """
    num_layers = len(model.layers)
    shapes: dict[tuple, list[int]] = {}
    blocks = [block_values(block) for block in model.layers]
    for i, block in enumerate(model.layers):
        shapes.setdefault((type(block), tuple(name for name, _, _ in blocks[i])), []).append(i)
    targets = [(None, name, host, prop) for name, host, prop in available(model)]
    for first, *_ in shapes.values():
        targets += [(first, name, host, prop) for name, host, prop in blocks[first]]
    try:
        seen = probe(model, targets, model.scan)
    except Exception:
        seen = probe(model, targets, model.trace)

    table: dict[int, dict[str, int]] = {-1: {}, num_layers: {}}
    steps: dict[int | None, dict] = {}  # location -> rank, per block and one for the root (None)
    root_side = -1  # becomes num_layers once the forward has reached a block
    for layer, name, location in seen:
        if layer is None:
            side = root_side
        else:
            side, root_side = layer, num_layers
        step = steps.setdefault(layer, {})
        table.setdefault(side, {})[name] = step.setdefault(location, len(step))
    for first, *rest in shapes.values():
        table.update((i, table[first]) for i in rest)
    return table


def cached(model: Any) -> dict[int, dict[str, int]]:
    """`compute`, once per model and kernel binding: `route_kernels` and `chunk_per_token` change what fires, so they invalidate it."""
    generation, table = model._order or (None, None)
    if generation != recurrent.generation:
        table = compute(model)
        model._order = (recurrent.generation, table)
    return table
