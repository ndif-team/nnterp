---
title: Head-selective K/V patching
one_liner: "Make chosen query heads read another run's keys and values over a span, on the last query row only, with `self_attn.interface`: the other heads of their group, the other rows and the KV cache stay as the model computed them."
tags: [patterns, attention, heads, patching, keys, values, grouped-query, generation]
related: [docs/usage/attention-interior.md, docs/patterns/activation-patching.md, docs/patterns/attention-patterns.md, docs/usage/generation.md]
sources: [nnterp/components/attention.py, tests/families/suite.py]
---

# Head-selective K/V patching

## What this is for

Activation patching on a head's *inputs*: does query head `h` of block `L`, reading the
keys and values another prompt has over a span, carry that prompt's information into the
prediction? The edit has to reach the chosen heads and nothing else, and writing
`attention_keys` or `attention_values` cannot do that:

- **The keys and values are `num_kv_heads` wide.** Under grouped-query attention
  key/value head `g` is read by every query head `h` with `h // group == g`, so an edit of
  it reaches the whole group.
- **Every query row reads them.** An edit of the keys at the span changes the attention of
  every row from the span on, not only the last.
- **Under `generate` they are the cache.** On most families the attention receives the
  cache's own tensors, so an in-place edit stays in the cache for every later step, and
  an assignment, though it lasts one forward, still reaches the whole group and every row.

The recipe: clone the keys and values, replace the span for the chosen heads' key/value
heads, call `self_attn.interface` (the model's own attention function, with this call's
mask and scale) on the copies, and write only the chosen heads' last row of
`attention_head_outputs`.

## Canonical pattern

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("Qwen/Qwen2.5-0.5B", dispatch=True, attn_implementation="eager")
LAYER, HEADS = 14, [9]                                    # the query heads that read the source
attn = model.layers[LAYER].self_attn
group = attn.num_heads // attn.num_kv_heads              # 7 query heads per key/value head
kv_heads = sorted({h // group for h in HEADS})           # [1]: the key/value head they read

source, target = "The capital of France is", "The capital of Italy is"
ids = model.tokenizer([source, target]).input_ids
span = [i for i, (a, b) in enumerate(zip(*ids)) if a != b]   # [3]: the country, at the same position in both

with model.trace(source):
    k_src = attn.attention_keys.save()                   # [1, num_kv_heads, seq, head_dim], rotated for their positions
    v_src = attn.attention_values.save()

def patch():
    """In a trace or a generate step: the chosen heads' last row, attended over the source's span."""
    q, k, v = attn.attention_queries, attn.attention_keys, attn.attention_values
    k2, v2 = k.clone(), v.clone()                        # copies: k and v are what the model attends with
    k2[:, kv_heads, span] = k_src[:, kv_heads, span]
    v2[:, kv_heads, span] = v_src[:, kv_heads, span]
    heads, _ = attn.interface(q, k2, v2)                 # the model's own attention function on the patched copies
    attn.attention_head_outputs[:, -1, HEADS] = heads[:, -1, HEADS]   # the chosen heads, the last row: the only write

with model.trace(target):
    clean = model.logits[:, -1].save()
with model.trace(target):
    patch()
    patched = model.logits[:, -1].save()

paris, rome = (model.tokenizer(w, add_special_tokens=False).input_ids[0] for w in (" Paris", " Rome"))
gap = lambda logits: round((logits[0, paris] - logits[0, rome]).item(), 3)
gap(clean), gap(patched)                                 # (-4.188, -4.062): Paris minus Rome, clean and patched
```

`patch()` reads in forward order: the queries, keys and values and `interface` at the
attention call, then the head outputs after it. The call reruns the whole attention
(every head, every row) on the copies, so its matmuls have the model's shapes and every
head outside the patched group comes out bit for bit as the model computed it; the
write then takes the chosen heads' last row alone. The span here is one position in
prompts of one length, so the source's keys, rotated for their positions, are at the
right positions in the target too.

## Why not write the keys

Written directly, the same span edit reaches every head of the group and every row
from the span on:

```python
with model.trace(target):
    served = attn.attention_head_outputs.save()          # [1, seq, num_heads, head_dim], unpatched
with model.trace(target):
    attn.attention_keys[:, kv_heads, span] = k_src[:, kv_heads, span]
    attn.attention_values[:, kv_heads, span] = v_src[:, kv_heads, span]
    bled = attn.attention_head_outputs.save()

moved = bled.ne(served).any(-1)[0]                       # [seq, num_heads]
moved[-1].nonzero().flatten().tolist()                   # [7, 8, 9, 10, 11, 12, 13]: every head of the group
moved[:-1].any().item()                                  # True: and the rows before the last
```

## Under `generate`

The same `patch()` on every step: the last row is the new token's, and on a decode step
the keys and values the call receives are the whole cache, the span among them. The
cache keeps the target's own keys, since nothing writes them:

```python
with model.generate(target, max_new_tokens=4, do_sample=False) as tracer:
    for step in tracer.iter[:4]:
        cached = attn.attention_keys[:, kv_heads].save() # this step's cache, after the earlier steps' patches
        patch()                                          # every step: the last row is the new token's
    out = tracer.result.save()

with model.trace(target):
    own = attn.attention_keys[:, kv_heads].save()
n = len(ids[1])
torch.equal(cached[:, :, :n], own)                       # True: the cache holds the target's own keys
```

`cached` is read before `patch()`: the keys come before the call, and `patch()` ends past
it, at the head outputs.

## Variations

- **Several heads, several blocks.** `HEADS` may hold any heads of the block; for several
  blocks, run `patch()`'s body per block in forward order, each with its block's
  `attention_keys` / `attention_values` from the source run.
- **Scale what the heads read.** `v2[:, kv_heads, span] = s * v_src[:, kv_heads, span]`
  amplifies the transplanted values without changing the pattern.
- **Every row, not the last.** Write `attention_head_outputs[:, :, HEADS] = heads[:, :, HEADS]`
  on the prompt pass.
- **Other positions.** The keys are served after the rotary embedding, rotated for the
  positions they were computed at; a span written at other positions in the target needs
  keys rotated for those positions.

## Gotchas

- **Clone before editing.** `k` and `v` are the tensors the model attends with and, under
  `generate`, the cache's; an in-place edit of them is the bleed above.
- **Read `interface` with the queries and call it before the head outputs.** The call
  takes the function as the forward runs it, so it comes before any read of the scores,
  the pattern or the head outputs; to call it twice, read `attn.interface` once and call
  the result twice ([attention-interior.md](../usage/attention-interior.md#calling-the-attention-function-again-interface)).
- **Pass whole tensors.** The call takes the live call's query and key lengths (its mask
  has them): patch inside `k2` and `v2`, pass all of `q`, and slice the result.
- **Eager only, and not everywhere.** `support()` reports `self_attn.interface` unavailable
  under `sdpa` and on families with no such call (BLOOM, MPT, Falcon, XGLM, DeepSeek-V4);
  check it before the trace.
- **One invoke for bit-exactness.** Across invokes the call runs one invoke's rows where
  the model ran them all at once, so what it computes agrees with the model's own to
  rounding, not bit for bit; a batch of prompts in one invoke is bit for bit.

## Related

- [attention-interior.md](../usage/attention-interior.md#calling-the-attention-function-again-interface): `interface` itself, and the values it reads and returns.
- [activation-patching](activation-patching.md): patching the stream or a contribution.
- [attention-patterns](attention-patterns.md): what the heads attend to before you patch them.
- [generation.md](../usage/generation.md): the values per decode step.
