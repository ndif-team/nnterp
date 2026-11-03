"""The forward order of a standardized model's values, measured by one probe run (see `probe`)."""

from __future__ import annotations

from typing import Any

import nnsight

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


TEXT = "The quick brown fox"  # any short text: the order does not depend on it


def prompts(model: Any) -> list[tuple[tuple, dict]]:
    """The inputs a probe may run on, each ``(args, kwargs)`` for an invoke, tried in order: on a model loaded
    with a processor, `TEXT` with one small image first (the processor's chat template puts the image's
    tokens in, or its ``image_token`` does where it has no template), so that a value an image gates (a block
    that adds a deepstack feature at the image tokens) is reached too; then `TEXT` alone, which is all a
    text-only load takes."""
    processor = getattr(model, "processor", None)
    token = getattr(processor, "image_token", None)
    found: list[tuple[tuple, dict]] = []
    if processor is not None and isinstance(token, str):
        from PIL import Image

        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": TEXT}]}]
        try:
            text = processor.apply_chat_template(messages, tokenize=False)
        except Exception:
            text = f"{token} {TEXT}"
        found.append(((text,), {"images": [Image.new("RGB", (64, 64), "white")]}))
    return found + [((TEXT,), {})]


def probe(model: Any, targets: list[tuple[int | None, str, Any, Any]], run: Any,
          prompt: tuple[tuple, dict] = ((TEXT,), {})) -> list[tuple[int | None, str, Any]]:
    """``(layer, name, location)`` for each target, in the order the forward reached them.

    One ``run`` (``model.scan``, meta tensors and no weights needed, or
    ``model.trace``) with one invoke for a prompt (`prompts`) and one empty
    invoke per value. Each empty invoke reads its value, which parks it until
    the forward reaches that value's location, then appends the location it
    was served at, so the list comes out in forward order. Workers parked on
    one location resume in the order they were written, not in any order the
    forward gives them, so values served at one location (the attention call's
    queries, keys and values) tie: either read order is legal. A derived value,
    computed from several locations, is its own location ``(layer, name)``.
    """
    args, kwargs = prompt
    with run() as tracer:
        seen = nnsight.save([])
        with tracer.invoke(*args, **kwargs):
            pass
        for layer, name, host, prop in targets:
            with tracer.invoke():
                getattr(host, prop.name)  # parks until the forward reaches it; nothing to save
                # Resolved after the read, when any drill is built.
                location = (layer, name) if isinstance(prop, DerivedEProperty) else prop._resolve(host, prop.path(host))
                seen.append((layer, name, location))
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
    A model loaded with its processor is probed with an image in the prompt
    (`prompts`), since a value an image gates (Qwen3-VL's ``deepstack_output``)
    is reached by no text-only forward; where that prompt cannot run, the
    text alone is tried.
    """
    num_layers = len(model.layers)
    shapes: dict[tuple, list[int]] = {}
    blocks = [block_values(block) for block in model.layers]
    for i, block in enumerate(model.layers):
        shapes.setdefault((type(block), tuple(name for name, _, _ in blocks[i])), []).append(i)
    targets = [(None, name, host, prop) for name, host, prop in available(model)]
    for first, *_ in shapes.values():
        targets += [(first, name, host, prop) for name, host, prop in blocks[first]]
    seen, failed = None, None
    for run in (model.scan, model.trace):
        for prompt in prompts(model):
            try:
                seen = probe(model, targets, run, prompt)
                break
            except Exception as e:      # a forward meta tensors cannot run, or a value the prompt never reaches
                failed = e
        if seen is not None:
            break
    if seen is None:
        raise failed if failed is not None else RuntimeError("the probe ran on nothing")

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
    """`compute`, once per model: route kernels before the model runs, so what fires never changes after."""
    if model._order is None:
        model._order = compute(model)
    return model._order
