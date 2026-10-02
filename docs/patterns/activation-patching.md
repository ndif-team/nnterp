---
title: Activation Patching
one_liner: "Patch `layer_output`, `attention_output` or `mlp_output` from a clean prompt into a corrupt one at a position, from a saved value or across invokes with a barrier, and sweep the layers; the contribution identity makes a sublayer patch mean the same thing on every family."
tags: [patterns, patching, causal, residual-stream]
related: [docs/usage/residual-stream.md, docs/patterns/ablation.md, docs/patterns/logit-lens.md, docs/patterns/delta-net-state.md, docs/patterns/cross-family-sweep.md]
sources: [nnterp/components/layer.py, nnterp/components/attention.py, nnterp/components/mlp.py, nnterp/standardized.py]
---

# Activation Patching

## What this is for

Activation patching asks whether a component *carries* the information that decides
the answer. Run a clean prompt and a corrupt one; copy one activation from the clean
run into the corrupt run at the same module and position; if the corrupt run now
gives the clean answer, that activation was sufficient.

The sites are the standard values. `layer_output` is the residual stream leaving a
block on every family, and `attention_output` / `mlp_output` are the sublayers'
contributions, defined by `layers[i].input + attention_output + mlp_output ==
layer_output`. So patching a contribution means exactly "the block adds the clean
sublayer's contribution instead of the corrupt one's, and nothing else changes",
on a sequential block, a parallel block, a sandwich-norm block and a hybrid's
DeltaNet block alike. That identity is what lets this page skip the per-architecture
tuple handling.

## Canonical pattern

Save the clean value in one trace, patch it in a second, with the unpatched corrupt
run as a second invoke of that trace:

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", dispatch=True)
clean = "The Eiffel Tower is in the city of"
corrupt = "The Colosseum is in the city of"
LAYER, POS = model.num_layers // 2, -1
ids = model.tokenizer(" Paris", add_special_tokens=False).input_ids
assert len(ids) == 1, model.tokenizer.convert_ids_to_tokens(ids)   # one token, or paris is not the word
paris = ids[0]

with model.trace(clean):
    clean_resid = model.layers[LAYER].layer_output.save()           # [1, seq, hidden]

with model.trace() as tracer:
    with tracer.invoke(corrupt):
        baseline = model.next_token_probs.save()
    with tracer.invoke(corrupt):
        model.layers[LAYER].layer_output[:, POS] = clean_resid[:, POS]
        patched = model.next_token_probs.save()

print(f"P(Paris)  corrupt {baseline[0, paris]:.3f}   patched {patched[0, paris]:.3f}")
```

The write is in place on the served tensor, so it reaches the model, and it touches
only its invoke's rows. `clean_resid` is an ordinary tensor by the time the second
trace runs, so nothing has to be synchronized.

`add_special_tokens=False` keeps the BOS token out of the target (`tokenizer.encode(" Paris")[0]`
is the BOS id on Llama and Gemma, and every probability then reads 0.000). The assertion
catches a word that is more than one token: Mistral's sentencepiece tokenizer gives
`['▁', '▁Paris']` and Granite's `['ĠPar', 'is']`. Then try the word without the leading
space (`"Paris"` is `['▁Paris']` on Mistral), or pick a target word that is one token.

## The ordering rule

Within one invoke, reads and writes follow the forward: a source read at block 3
can feed a write at block 5, not the reverse, and a block's own clean value cannot
come from the same invoke. Across invokes of one trace, a write that *uses* another
invoke's value needs a barrier, because the assignment evaluates its right-hand
side before it parks the worker; without one the name is not bound yet and the
line raises `NameError`:

```python
with model.trace() as tracer:
    barrier = tracer.barrier(2)
    with tracer.invoke(clean):
        source = model.layers[LAYER].layer_output[:, POS]
        barrier()                                                # source is read
    with tracer.invoke(corrupt):
        barrier()                                                # wait for it
        model.layers[LAYER].layer_output[:, POS] = source
        patched = model.next_token_probs.save()
```

This gives the same probabilities as the two-trace form. A session is the third
option, one scope over several traces with no `.save()` between them:

```python
with model.session():
    with model.trace(clean):
        source = model.layers[LAYER].layer_output[:, POS]
    with model.trace(corrupt):
        model.layers[LAYER].layer_output[:, POS] = source
        patched = model.next_token_probs.save()
```

Two traces with saves is the form that also runs remotely and needs no ordering
thought; use the barrier when the clean and corrupt rows must share one forward.
See nnsight `docs/usage/barrier.md` and `docs/usage/session.md`.

## Variations

### Sweep the layers

Cache the clean run once, one trace per layer:

```python
cache = []                                   # made outside the trace
with model.trace(clean):
    for layer in model.layers:
        cache.append(layer.layer_output.save())

sweep = []
for i in range(model.num_layers):
    with model.trace(corrupt):
        model.layers[i].layer_output[:, POS] = cache[i][:, POS]
        sweep.append(model.next_token_probs[0, paris].save())

for i, p in enumerate(sweep):
    print(f"layer {i:2d}: P(Paris) {float(p):.3f}")
```

Two ends of this curve are fixed by construction, not by the prompt: patching the
last block at the last position reproduces the clean prediction exactly, and
patching an early block at a non-final position overwrites everything computed
there so far. Read a residual sweep as *reach*, and localize with the map.

### Layer x position map

One trace per layer, one invoke per position. The two prompts must tokenize to the
same length for positions to correspond; check it, since the tokenizer decides:

```python
n = len(model.tokenizer(corrupt).input_ids)
assert n == len(model.tokenizer(clean).input_ids)

grid = []
for i in range(model.num_layers):
    row = []                                 # outside the trace
    with model.trace() as tracer:
        for pos in range(n):
            with tracer.invoke(corrupt):
                model.layers[i].layer_output[:, pos] = cache[i][:, pos]
                row.append(model.next_token_probs[0, paris].save())
    grid.append([float(p) for p in row])     # [layers][positions]
```

### Patch a contribution instead of the stream

```python
with model.trace(clean):
    clean_attn = model.layers[LAYER].self_attn.attention_output.save()
    clean_mlp = model.layers[LAYER].mlp.mlp_output.save()

with model.trace(corrupt):
    model.layers[LAYER].self_attn.attention_output[:, POS] = clean_attn[:, POS]
    patched_attn = model.next_token_probs.save()

with model.trace(corrupt):
    model.layers[LAYER].mlp.mlp_output[:, POS] = clean_mlp[:, POS]
    patched_mlp = model.next_token_probs.save()
```

Patching `layer_output` at block `i` replaces everything up to and including `i`
at that position; patching a contribution replaces one sublayer's addition. On a
hybrid the DeltaNet block's contribution is `linear_attn.attention_output`, and its
recurrent state is patchable too: see [delta-net-state](delta-net-state.md).

### One head

`attention_head_outputs[..., HEAD, :]` is one head's output before the projection,
`[batch, seq, head_dim]`; patch that slice the same way, under
`attn_implementation="eager"`.

### Noising

Swap the roles: run the clean prompt and paste the corrupt activation in. The code
is symmetric.

## Interpretation

- Keep an unpatched corrupt baseline in the same trace, as above; batch effects
  shift probabilities slightly and the baseline is the reference.
- Position is the question. The subject's position localizes the *source* of a
  fact; the last position tests whether the *final prediction* is sensitive.
- Compare a logit difference (`Paris` minus `Rome`) rather than the top token when
  the corrupt run is near-degenerate; a tiny numerical shift can flip an argmax.
- Effects are often a few percent on one pair. Average over pairs.

## Gotchas

- Across invokes, a consuming *write* needs a barrier; without it, `NameError`.
  Across traces, `.save()` or a session.
- Within one invoke, reads follow the forward (`OutOfOrderError` otherwise), and
  the source cannot be the target block's own clean value.
- Different token lengths cannot share an absolute position; `[:, -1]` survives
  padding and length differences, an absolute index does not.
- Patching mutates the served tensor; `.clone()` first if you want both.
- A name bound inside the trace does not survive it; `cache`, `sweep`, `row` are
  made outside.
- On a family whose block returns a tuple (GPT-J, GPT-Neo, BLOOM, MPT,
  Falcon), `layer_output` is still the tensor, and an assignment puts it back in the tuple.
  No `[0]` indexing and no tuple rebuild.
- On Gemma-4 the KV-sharing blocks attend with an earlier block's keys and values.
  A patch of `attention_keys` / `attention_values`, in place or by assignment, is
  local to the block it is made on; to patch what every borrower attends with, patch
  the source block's `k_proj` / `v_proj` output
  ([Borrowed keys and values](../reference/families.md#borrowed-keys-and-values)).
- On Granite (and GraniteMoE, Granite-SWA, HyperCLOVA X, ZAYA) a patch of
  `attention_output` or `mlp_output` at one position moves the other positions by
  rounding; in bf16 that can be a visible fraction of a small patch's effect, so load in
  float32 ([families](../reference/families.md#scaled-residual-adds)).

## Related

- [ablation](ablation.md): zero instead of paste.
- [logit-lens](logit-lens.md): where the answer appears, before asking what carries it.
- [delta-net-state](delta-net-state.md): patching a hybrid's recurrent state.
- [cross-family-sweep](cross-family-sweep.md): the same sweep across checkpoints.
- [../usage/residual-stream.md](../usage/residual-stream.md): the contribution
  identity.
- nnsight `docs/usage/barrier.md`, `docs/usage/session.md`,
  `docs/usage/invoke-and-batching.md`.
- Meng et al. (2022), "Locating and Editing Factual Associations in GPT".
