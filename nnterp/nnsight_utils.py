"""Collecting activations and next-token distributions over many prompts.

Written against the standard values: a layer's activation is ``model.layers[i].layer_output`` unless a
``get_activations(model, layer)`` of your own says otherwise.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch.utils.data import DataLoader

from .standardized import StandardizedTransformer

GetActivations = Callable[[StandardizedTransformer, int], torch.Tensor]


def layer_output(model: StandardizedTransformer, layer: int) -> torch.Tensor:
    """The default ``get_activations``: the residual stream leaving block ``layer``."""
    return model.layers[layer].layer_output


def _check_index(model: StandardizedTransformer, idx: int) -> None:
    side = model.tokenizer.padding_side
    if idx < 0 and side != "left":
        raise ValueError(f"a negative token index needs left padding, and the tokenizer pads {side!r}")
    if idx > 0 and side != "right":
        raise ValueError(f"a positive token index needs right padding, and the tokenizer pads {side!r}")


@torch.no_grad()
def get_token_activations(
    model: StandardizedTransformer,
    prompts: str | list[str] | None = None,
    layers: list[int] | None = None,
    get_activations: GetActivations | None = None,
    remote: bool = False,
    idx: int | None = None,
    tracer=None,
) -> torch.Tensor:
    """The activation at one token position of each prompt, at each layer.

    Args:
        model: The standardized model.
        prompts: What to run; ``None`` when called inside an open ``tracer``.
        layers: Which blocks; default all.
        get_activations: ``(model, layer) -> tensor`` giving a ``[batch, seq, ...]``
            value; default `layer_output`.
        remote: Run on NDIF.
        idx: The token position; default ``-1``, the last token. A negative
            index needs left padding, a positive one right padding.
        tracer: An open trace to read from instead of opening one.

    Returns:
        ``[num_layers, num_prompts, hidden]`` on the CPU (still on the
        model's device when read inside a caller's ``tracer``).
    """
    if tracer is None and prompts is None:
        raise ValueError("prompts must be given when no tracer is")
    get_activations = get_activations or layer_output
    idx = -1 if idx is None else idx
    _check_index(model, idx)
    layers = list(range(model.num_layers)) if layers is None else layers
    acts = []
    if tracer is None:
        with model.trace(prompts, remote=remote) as tracer:
            for layer in layers:
                acts.append(get_activations(model, layer)[:, idx].cpu().save())
            tracer.stop()
    else:
        for layer in layers:
            acts.append(get_activations(model, layer)[:, idx])
    return torch.stack(acts)


@torch.no_grad()
def collect_last_token_activations_session(
    model: StandardizedTransformer,
    prompts: list[str],
    batch_size: int,
    layers: list[int] | None = None,
    get_activations: GetActivations | None = None,
    remote: bool = False,
    idx: int | None = None,
) -> torch.Tensor:
    """`get_token_activations` over batches inside one ``model.session``, so a remote run is one request."""
    get_activations = get_activations or layer_output
    idx = -1 if idx is None else idx
    _check_index(model, idx)
    layers = list(range(model.num_layers)) if layers is None else layers
    with model.session(remote=remote):
        all_acts = []
        for batch in DataLoader(prompts, batch_size=batch_size):
            with model.trace(batch) as tracer:
                acts = [get_activations(model, layer)[:, idx].cpu().save() for layer in layers]
                tracer.stop()
            all_acts.append(torch.stack(acts).save())
        all_acts = torch.cat(all_acts, dim=1).save()
    return all_acts


def collect_token_activations_batched(
    model: StandardizedTransformer,
    prompts: list[str],
    batch_size: int,
    layers: list[int] | None = None,
    get_activations: GetActivations | None = None,
    remote: bool = False,
    idx: int | None = None,
    tqdm=None,
    use_session: bool = True,
) -> torch.Tensor:
    """`get_token_activations` over batches; ``[num_layers, num_prompts, hidden]`` on the CPU.

    A remote run goes through `collect_last_token_activations_session` (one
    request) unless ``use_session`` is off. ``tqdm`` is a progress-bar factory
    to wrap the batch loop with, or ``None``.
    """
    if use_session and remote:
        return collect_last_token_activations_session(model, prompts, batch_size, layers, get_activations, remote, idx)
    steps = range(0, len(prompts), batch_size)
    acts = [
        get_token_activations(model, prompts[i : i + batch_size], layers, get_activations, remote, idx)
        for i in (tqdm(steps) if tqdm is not None else steps)
    ]
    return torch.cat(acts, dim=1)


def compute_next_token_probs(model: StandardizedTransformer, prompt: str | list[str], remote: bool = False) -> torch.Tensor:
    """The next-token distribution of each prompt, ``[num_prompts, vocab]`` on the CPU."""
    with model.trace(prompt, remote=remote):
        probs = model.next_token_probs.cpu().save()
    return probs
