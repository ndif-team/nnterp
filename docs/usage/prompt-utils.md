---
title: Prompt Utils
one_liner: Track named sets of target tokens in the next-token distribution over many prompts with `nnterp.prompt_utils`.
tags: [usage, prompts, tokens, next-token, batching]
related: [docs/usage/root-values.md, docs/usage/methods.md, docs/usage/loading.md, docs/usage/activations.md, docs/usage/generation.md, docs/usage/remote.md]
sources: [nnterp/prompt_utils.py, nnterp/nnsight_utils.py, nnterp/standardized.py]
---

# Prompt Utils

## What this is for

`nnterp.prompt_utils` answers one question at scale: how much probability does
the model put on *these* words next? `get_first_tokens` turns words into the
token ids a model would emit for them, `Prompt` pairs a prompt with named sets
of those ids, and `run_prompts` runs many prompts in batches and returns each
target's probability mass per prompt. Everything reads
`model.next_token_probs`, a standard value, so one script runs on every family.

## Canonical pattern

```python
from nnterp import StandardizedTransformer
from nnterp.prompt_utils import Prompt, run_prompts

model = StandardizedTransformer("openai-community/gpt2")

prompts = [
    Prompt.from_strings("The capital of France is", {"correct": "Paris", "wrong": ["London", "Berlin"]}, model),
    Prompt.from_strings("The capital of Germany is", {"correct": "Berlin", "wrong": ["London", "Paris"]}, model),
]
mass = run_prompts(model, prompts, batch_size=32)
mass["correct"].shape      # torch.Size([2, 1]): [num_prompts, layers]; the next-token distribution is one "layer"
mass["wrong"][0, 0]        # the mass on ' London' and ' Berlin' (and 'London', 'Berlin') after the first prompt
```

## Words to token ids: `get_first_tokens`

```python
get_first_tokens(words: str | list[str], model_or_tokenizer, use_hacky_implementation=False) -> list[int]
```

For each word, the first token of `word` and the first token of `" word"`,
deduplicated in order (the `" word"` id is dropped when it is only the space
token). Given a model, it tokenizes with `model.add_prefix_false_tokenizer`,
the checkpoint's tokenizer loaded with `add_prefix_space=False`, so `"Paris"`
and `" Paris"` are different tokens. Given a tokenizer that adds a prefix
space, the two collide; the function then warns and falls back to the pear
trick (tokenize `"🍐" + word` and drop the pear), which
`use_hacky_implementation=True` forces. `TokenizationError` is raised when the
trick does not tokenize as expected either.

```python
ids = get_first_tokens(["Paris", "London"], model)
[model.tokenizer.decode([i]) for i in ids]     # what the model would emit for each: check this before a run
```

## `Prompt`

A dataclass: `prompt` (the text), `target_tokens` (`{name: [token ids]}`) and
`target_strings` (what those came from, when built with `from_strings`).

```python
Prompt.from_strings(prompt, target_strings, model_or_tokenizer) -> Prompt
```

`target_strings` is a dict of name to a word or list of words; a bare string
or list is one target named `"target"`. Each name's words go through
`get_first_tokens`.

```python
prompt.has_no_collisions(ignore_targets=None) -> bool
```

Whether no token id belongs to two targets, leaving out the names in
`ignore_targets`. A shared id is counted in both targets' mass, so check this
when targets are words that can start alike.

```python
prompt.get_target_probs(probs, layer=None) -> dict[str, Tensor]
```

`probs` is `[batch, layers, vocab]`; each target's mass is the sum over its
ids, `[batch, layers]` on the CPU, or `[batch]` for one `layer`.

```python
prompt.run(model, get_probs) -> dict[str, Tensor]
```

`get_probs(model, prompt.prompt)` returns `[batch, layers, vocab]`; the result
is `get_target_probs` of it. The default `get_probs` is
`next_token_probs_unsqueeze(model, prompt, remote=False)`:
`compute_next_token_probs` with a layer axis of one, `[batch, 1, vocab]`.

## Many prompts: `run_prompts`

```python
run_prompts(model, prompts, batch_size=32, get_probs_func=None, func_kwargs=None, remote=False, tqdm=None) -> dict[str, Tensor]
```

Runs the prompts' texts in batches of `batch_size`, one trace per batch, under
`torch.no_grad()`. `get_probs_func(model, batch, remote=..., **func_kwargs)`
returns `[batch, layers, vocab]`; the default is the next-token distribution
with one layer. The result maps each target name to `[num_prompts, layers]` on
the CPU. Every prompt must name the same targets (`ValueError` otherwise); an
empty list returns `{}`. `tqdm` is a progress-bar factory to wrap the batch
loop with (`tqdm=tqdm.tqdm`), or `None`.

A `get_probs_func` of your own turns the layer axis into something: the logit
lens at chosen blocks, through `model.project_on_vocab`:

```python
def lens_probs(model, batch, remote=False, layers=None):
    with model.trace(batch, remote=remote):
        rows = [model.project_on_vocab(model.layers[i].layer_output)[:, -1].softmax(-1) for i in layers]
        out = torch.stack(rows, dim=1).cpu().save()          # [batch, len(layers), vocab]
    return out

mass = run_prompts(model, prompts, batch_size=32, get_probs_func=lens_probs, func_kwargs={"layers": [4, 8]})
mass["correct"].shape      # torch.Size([2, 2]): the target's mass at block 4 and block 8
```

## Gotchas

- **Decode `target_tokens` before trusting a run.** A first token is whatever the
  tokenizer makes of the word: on a byte-pair vocabulary a rare word's first
  token is a fragment (`'Par'`), and the mass then counts every continuation.
- **The result is per target, not per word.** A target with several words sums
  their ids; `has_no_collisions()` says whether two targets share one.
- **Batched runs need left padding.** `next_token_probs` is the last position
  of every row, which is each prompt's last token only under left padding. Load
  with `tokenizer_kwargs={"padding_side": "left"}` when a tokenizer pads right.
- **`get_first_tokens` with a raw tokenizer warns and changes strategy** when the
  tokenizer adds a prefix space; pass the model, whose
  `add_prefix_false_tokenizer` keeps `"word"` and `" word"` apart.
- **`remote=True` passes through** to every trace; `get_probs_func` receives it as
  a keyword and must forward it to `model.trace`. See [remote.md](remote.md).

## Related

- [root-values.md](root-values.md), `next_token_probs` and `add_prefix_false_tokenizer`.
- [methods.md](methods.md), `project_on_vocab` for a layer-wise `get_probs_func`.
- [loading.md](loading.md), `tokenizer_kwargs` and the padding side.
- [activations.md](activations.md), `nnterp.nnsight_utils`: `compute_next_token_probs` is what the default `get_probs` calls.
- [generation.md](generation.md), the same `next_token_probs` at every step of `generate`.
- [remote.md](remote.md), running these on NDIF.
