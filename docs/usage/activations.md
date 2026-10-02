---
title: Collecting Activations
one_liner: Gather one token position's activation at chosen blocks over many prompts, `[num_layers, num_prompts, hidden]`, with `nnter.nnsight_utils`.
tags: [usage, activations, batching, residual-stream]
related: [docs/usage/residual-stream.md, docs/usage/loading.md, docs/usage/prompt-utils.md, docs/usage/generation.md, docs/usage/remote.md]
sources: [nnter/nnsight_utils.py, nnter/standardized.py, nnter/components/layer.py]
---

# Collecting Activations

## What this is for

Probing, steering vectors and logit lenses all start the same way: the
residual stream at one position of every prompt, at the blocks you care about,
stacked into one tensor. `nnter.nnsight_utils` does that against the standard
values, so the same call runs on every family: `get_token_activations` for one
batch, `collect_token_activations_batched` over many prompts, and
`compute_next_token_probs` for the distribution at the end. The names are
nnterp's.

## Canonical pattern

```python
from nnter import StandardizedTransformer
from nnter.nnsight_utils import get_token_activations

model = StandardizedTransformer("openai-community/gpt2", tokenizer_kwargs={"padding_side": "left"})

acts = get_token_activations(model, ["The Eiffel Tower is in", "Hello"], layers=[4, 8])
acts.shape     # torch.Size([2, 2, 768]): [num_layers, num_prompts, hidden], on the CPU
```

`acts[i, j]` is `model.layers[layers[i]].layer_output[j, -1]`: the residual
stream leaving block `layers[i]` at the last token of prompt `j`.

## `get_token_activations`

```python
get_token_activations(model, prompts=None, layers=None, get_activations=None, remote=False, idx=None, tracer=None) -> Tensor
```

- `prompts`: a string or list; `None` only with `tracer`.
- `layers`: which blocks; default every block, in order.
- `get_activations(model, layer) -> Tensor`: the `[batch, seq, ...]` value to
  read at block `layer`; default `layer_output`, which is
  `model.layers[layer].layer_output`. Any standard value with a sequence axis
  works: `lambda m, i: m.layers[i].self_attn.attention_output`.
- `idx`: the token position, default `-1`. A negative index needs left padding
  and a positive one right padding, checked against
  `model.tokenizer.padding_side` before the trace and raised as `ValueError`;
  `0` needs neither.
- `remote`: run on NDIF ([remote.md](remote.md)).
- `tracer`: an open trace to read from instead of opening one. The result is
  then still on the model's device, and yours to `.save()`:

```python
with model.trace(prompts) as tracer:
    acts = get_token_activations(model, layers=[4, 8], tracer=tracer).save()
```

Returns `[num_layers, num_prompts, hidden]` on the CPU (the trace stops after
the last requested block, so blocks past it do not run).

## Over many prompts

```python
collect_token_activations_batched(model, prompts, batch_size, layers=None, get_activations=None, remote=False, idx=None, tqdm=None, use_session=True) -> Tensor
```

`get_token_activations` over `prompts` in batches of `batch_size`, concatenated
along the prompt axis: `[num_layers, num_prompts, hidden]` on the CPU. `tqdm`
is a progress-bar factory to wrap the batch loop with, or `None`. With
`remote=True` and `use_session` on, it goes through
`collect_last_token_activations_session` instead, so one request carries every
batch.

```python
collect_last_token_activations_session(model, prompts, batch_size, layers=None, get_activations=None, remote=False, idx=None) -> Tensor
```

The same batches inside one `model.session(remote=remote)`: the traces share a
scope, and a remote run is one round trip (nnsight docs/usage/session.md).
Same result and shape.

```python
compute_next_token_probs(model, prompt, remote=False) -> Tensor
```

`model.next_token_probs` for a string or list of prompts, `[num_prompts, vocab]`
on the CPU. Under left padding the row is each prompt's own last token.

## Gotchas

- **The default `idx=-1` needs left padding.** With a tokenizer that pads right
  the call raises `ValueError("a negative token index needs left padding, and
  the tokenizer pads 'right'")` before anything runs; load with
  `tokenizer_kwargs={"padding_side": "left"}`. A positive `idx` needs right
  padding for the same reason.
- **The layer axis follows `layers`, not block numbers.** `layers=[4, 8]` gives
  rows 0 and 1.
- **`get_activations` must return a `[batch, seq, ...]` value.** The helper
  indexes `[:, idx]` on it; a value without a sequence axis in position 1
  (softmax attention's queries, keys and values put it at 2) needs a lambda
  that reshapes first.
- **The `tracer=` form does not `.save()` for you** and does not stop the trace;
  the result is a live tensor on the model's device until you save it.
- **`remote=True` batches change nothing about shapes**, only where the traces
  run; without a session each batch is its own request.

## Related

- [residual-stream.md](residual-stream.md), `layer_output`, the default value collected.
- [loading.md](loading.md), `tokenizer_kwargs` and the padding side.
- [prompt-utils.md](prompt-utils.md), probability mass on target tokens, built on `compute_next_token_probs`.
- [generation.md](generation.md), reading `layer_output` per decode step instead of per prompt.
- [remote.md](remote.md), what `remote=True` needs from the server.
