---
title: Attention Patterns
one_liner: "Read `attention_probabilities` as `[batch, heads, query, key]` on any family loaded eager, score heads for entropy, previous-token and first-token behavior, and edit the pattern in place."
tags: [patterns, attention, heads, source]
related: [docs/usage/availability.md, docs/usage/loading.md, docs/patterns/ablation.md, docs/patterns/contribution-decomposition.md, docs/patterns/logit-lens.md]
sources: [nnter/components/attention.py, nnter/components/eproperty.py, nnter/families/gpt_oss.py]
---

# Attention Patterns

## What this is for

The attention pattern is the matrix the values are mixed with: row `i` says how
much position `i` reads from each earlier position, per head. It is the most direct
read on what a head does, and the first place to look for previous-token heads,
induction heads and attention sinks.

`model.layers[i].self_attn.attention_probabilities` is that matrix on every family
that runs transformers' eager attention, `[batch, heads, query, key]`, read after the
softmax and the dropout: in the model's dtype, and with an attention sink's column
already dropped. It reaches inside the forward through nnsight `.source`, so it needs
`attn_implementation="eager"`. Those two facts are the whole per-family story; the
rest of this page is the same code on every checkpoint.

## Canonical pattern

```python
import torch
from nnter import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", dispatch=True, attn_implementation="eager")
prompt = "The cat sat on the mat because the cat"
attention_blocks = [i for i, layer in enumerate(model.layers) if getattr(layer, "self_attn", None) is not None]
LAYER = attention_blocks[len(attention_blocks) // 2]      # num_layers // 2 can be a linear block on a hybrid

with model.trace(prompt):
    pattern = model.layers[LAYER].self_attn.attention_probabilities.save()   # [batch, heads, query, key]

heads = pattern[0]                                                # [heads, query, key]: one heatmap per head
tokens = [model.tokenizer.decode(t) for t in model.tokenizer(prompt).input_ids]   # axis labels, both axes

assert pattern.shape[1] == model.num_heads
assert torch.equal(pattern.tril(), pattern)                      # causal: nothing above the diagonal
assert torch.allclose(pattern.sum(-1), torch.ones_like(pattern.sum(-1)), atol=1e-4)   # rows sum to one (see the sink caveat)
```

Without `attn_implementation="eager"` the value is unavailable and `support()` says
why, before anything runs:

```python
sdpa = StandardizedTransformer("openai-community/gpt2", dispatch=True)         # the checkpoint's default is sdpa
sdpa.support()["self_attn.attention_probabilities"]
# {0: "read inside the eager attention forward, but this model runs 'sdpa'; load with attn_implementation='eager'", 1: ...}
```

Reading it anyway raises `nnter.Unavailable` with the same reason, at that line.

## Per-head metrics

All three are one line on the saved tensor, `[batch, heads]` each:

```python
entropy = -(pattern * (pattern + 1e-12).log()).sum(-1).mean(-1)      # low: sharp heads
previous = pattern.diagonal(offset=-1, dim1=-2, dim2=-1).mean(-1)     # mass on key i-1: previous-token heads
first = pattern[..., 1:, 0].mean(-1)                                   # mass on key 0 from every later query: sink heads
```

A head with `previous > 0.5` attends mostly to the token before; a head with
`first > 0.5` parks most of its mass on the first token, which usually means it is
not engaged on this prompt. An induction head attends from `i` to the token after
the previous occurrence of the token at `i`; build that target from the token ids
and take the same mean.

### Every attention block at once

A hybrid (Qwen3.5) has `self_attn` on one block in four and `linear_attn` on the
rest, with no pattern, and a pure state-space model (Mamba) has none at all. Decide which
blocks have one *outside* the trace, as `attention_blocks` above does, then stack:

```python
with model.trace(prompt):
    patterns = torch.stack([model.layers[i].self_attn.attention_probabilities
                            for i in attention_blocks]).save()      # [blocks, batch, heads, query, key]
```

On a non-hybrid `attention_blocks` is every block; the same code runs. The
patterns are read in block order, as one trace requires.

## Editing the pattern

The value is the tensor the values are mixed with, so a write reaches the model.
Zero one head's pattern, and the head reads nothing:

```python
HEAD = 0
with model.trace(prompt):
    model.layers[LAYER].self_attn.attention_probabilities[:, HEAD] = 0
    logits = model.logits.save()
```

A uniform causal pattern, assigned:

```python
with model.trace(prompt):
    pattern = model.layers[LAYER].self_attn.attention_probabilities
    uniform = torch.ones_like(pattern).tril()
    model.layers[LAYER].self_attn.attention_probabilities = uniform / uniform.sum(-1, keepdim=True)
    logits = model.logits.save()
```

Knock out attention *to* one key position, then renormalize (or leave the mass
removed, which is a different question):

```python
KEY = 0
with model.trace(prompt):
    pattern = model.layers[LAYER].self_attn.attention_probabilities
    pattern[..., KEY] = 0
    pattern /= pattern.sum(-1, keepdim=True).clamp_min(1e-12)
    logits = model.logits.save()
```

Zeroing a head's pattern and zeroing its slice of `attention_head_outputs` give
the same logits; see [ablation](ablation.md) for the head-level recipes.

## The scores

`attention_scores` is the masked, scaled input to the softmax, same layout;
`attention_scores.softmax(-1)` equals the pattern up to the cast to the model
dtype. Edit the scores to change *relative* attention without breaking
normalization: adding a constant to one key column shifts mass toward it.

The causal mask is already in the scores (the masked entries hold the dtype's minimum,
`-inf` on GPT-Neo), so an edit that overwrites them lifts it: `scores[:] = 0` or
`scores * 0` makes every query attend to future tokens, and on GPT-Neo `scores * 0` is
NaN. Additive edits keep the mask. To set the real entries, keep the masked ones:

```python
with model.trace(prompt):
    scores = model.layers[LAYER].self_attn.attention_scores
    masked = scores <= torch.finfo(scores.dtype).min                  # the causal (and padding) mask
    model.layers[LAYER].self_attn.attention_scores = torch.where(masked, scores, torch.zeros_like(scores))
    logits = model.logits.save()                                      # uniform over the past, nothing above the diagonal
```

## The sink caveat

On GPT-OSS each head carries a learned sink logit that joins the softmax as an
extra key column and is dropped afterwards. The pattern is read after the drop, so
its rows sum to *less* than one, by the mass the sink took; the family's `Attention`
marks this with `SINK = True`. Entropy and the row-sum assertion above have to
account for it, and `attention_scores` on that family is read one step earlier,
at the masked scores before the sink column joins. The pattern's meaning, mass on
real keys, is unchanged.

Granite-SWA (`granite_swa`, `granitemoe_swa`) carries a learned sink too, but outside
the softmax: the pattern's rows sum to one, and the sink scales each head's *output* by
`sigmoid(logsumexp(scores) - sink)` after the values are mixed (between 0.009 and 0.95 on
one block of granite-swash-2b). So `attention_probabilities` there is not what reaches
`attention_head_outputs`, and forcing the pattern does not control a head's output; edit
`attention_head_outputs` instead. `SINK` is `False` on those families.

## Gotchas

- Eager is required and is not the default: a checkpoint loads `sdpa` unless you
  pass `attn_implementation="eager"`. Check `support()` rather than `hasattr`, which
  raises `Unavailable` when the value is unavailable.
- Reads follow the forward. In one trace read the pattern before `model.logits`;
  after a write to the pattern, do not read it back once the model has moved on
  (`OutOfOrderError`).
- A pattern is `heads * seq * seq` per block. Save the blocks and heads you need,
  index inside the trace, and wrap read-only capture in `torch.no_grad()`.
- On a batch of prompts the rows are left-padded; a padded row's pattern has zero
  columns at its pad positions and its real tokens start later. The pad *query* rows are
  not attention at all (uniform over every key on GPT-2, all on key 0 on GPT-Neo), so
  mask them with `model.attention_mask` before a per-row metric such as entropy.
- A pattern or scores read in two invokes of one trace raises `TypeError: 'NoneType'
  object is not subscriptable` today (an nnsight bug with values nnter reads inside a
  nested `.source` call). Read one prompt's pattern per trace, or the whole batch in one
  invoke; `attention_head_outputs` works across invokes.
- The pattern is in the model's dtype: on a bf16 checkpoint rows sum to one within
  a few ulps, not exactly.
- A GPT-2 checkpoint with `reorder_and_upcast_attn` set takes GPT-2's own upcast
  path, off the shared interface; `support()` reports the pattern unavailable there.
- GPT-J, GPT-Neo, BLOOM, MPT and Falcon compute attention themselves; their families map
  the pattern onto their own softmax (Falcon's onto the softmax without alibi and
  onto the dropout after the second softmax with it, by `config.alibi`).

## Related

- [ablation](ablation.md): removing a head via its pattern or its head outputs.
- [contribution-decomposition](contribution-decomposition.md): what each head
  writes, from `attention_head_outputs`.
- [logit-lens](logit-lens.md).
- [../usage/availability.md](../usage/availability.md): `support()` and the reasons.
- [../usage/loading.md](../usage/loading.md): `attn_implementation` at load.
- nnsight `docs/usage/source.md`: how `.source` reaches operations inside a forward.
