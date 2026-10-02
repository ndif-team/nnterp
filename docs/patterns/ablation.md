---
title: Ablation
one_liner: "Zero or mean-ablate `attention_output`, `mlp_output`, one head via `attention_head_outputs`, a DeltaNet mixer, or whole blocks via `skip_layers`, with the clean and ablated runs as two invokes of one trace measured on `next_token_probs`."
tags: [patterns, ablation, intervention, heads, hybrids]
related: [docs/usage/residual-stream.md, docs/usage/root-values.md, docs/usage/availability.md, docs/patterns/activation-patching.md, docs/patterns/attention-patterns.md, docs/patterns/cross-family-sweep.md]
sources: [nnter/standardized.py, nnter/components/attention.py, nnter/components/mlp.py, nnter/components/linear_attention.py, nnter/components/layer.py]
---

# Ablation

## What this is for

Ablation removes a component's output, or replaces it with a baseline, and measures
how the prediction moves. If zeroing a component destroys a behavior, the behavior
depends on it.

The components are the contribution values: `attention_output` and `mlp_output` are
*what the sublayer adds to the residual stream* on every family, including the ones
where the residual is added inside the module (BLOOM, MPT) or a post-sublayer norm
sits in the way (Gemma-2/3, OLMo-2). So "zero the MLP's contribution" is one
statement everywhere, and a hybrid's DeltaNet mixer has the same `attention_output`
as an attention block. That is the normalization this page relies on.

## Canonical pattern

Clean and ablated as two invokes of one trace, so both rows come from one forward,
measured on the next-token distribution:

```python
import torch
import torch.nn.functional as F
from nnter import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", dispatch=True, attn_implementation="eager")
prompt = "The Eiffel Tower is in the city of"
LAYER = model.num_layers // 2
ids = model.tokenizer(" Paris", add_special_tokens=False).input_ids
assert len(ids) == 1, model.tokenizer.convert_ids_to_tokens(ids)   # one token, or target is not the word
target = ids[0]

with model.trace() as tracer:
    with tracer.invoke(prompt):
        clean = model.logits[:, -1].float().log_softmax(-1).save()     # [1, vocab], log-probabilities
    with tracer.invoke(prompt):
        model.layers[LAYER].mlp.mlp_output[:] = 0                      # this invoke's rows only
        ablated = model.logits[:, -1].float().log_softmax(-1).save()

print(f"P(Paris)  clean {clean[0, target].exp():.3f}   ablated {ablated[0, target].exp():.3f}")
kl = F.kl_div(ablated, clean, log_target=True, reduction="none").sum(-1)   # KL(clean || ablated): the whole distribution's move
```

`add_special_tokens=False` keeps the BOS token out of the target (`tokenizer.encode(" Paris")[0]`
is the BOS id on Llama and Gemma, and every probability then reads 0.000). The assertion
catches a word that is more than one token: Mistral's sentencepiece tokenizer gives
`['▁', '▁Paris']` and Granite's `['ĠPar', 'is']`. Then try the word without the leading
space (`"Paris"` is `['▁Paris']` on Mistral), or pick a target word that is one token. The KL is taken on log-probabilities: `next_token_probs` underflows to exact zeros
(Pythia's float16 checkpoint has thousands per row), and `p * (p.log() - q.log())` is then
NaN.

`mlp_output` is a tensor, so `[:] = 0` writes in place and reaches the model. An
in-place write in one invoke touches only that invoke's rows; the clean row equals the
prompt run alone to float error (batched kernels reduce in another order, so compare
within one trace rather than with `torch.equal` across traces). Neither invoke reads a value the other produced, so
no barrier is needed.

## Variations

### The attention sublayer, one position

```python
with model.trace(prompt):
    model.layers[LAYER].self_attn.attention_output[:, -1] = 0
    probs = model.next_token_probs.save()
```

### Mean ablation

Replace the contribution with its mean over a reference set instead of zero, which
stays closer to the model's usual state:

```python
reference = ["The capital of France is", "Paris is in", "Berlin lies in", "London hosts the"]

with model.trace(reference):
    mean_act = model.layers[LAYER].mlp.mlp_output[:, -1].mean(0).save()   # [hidden]

with model.trace(prompt):
    model.layers[LAYER].mlp.mlp_output[:, -1] = mean_act
    probs = model.next_token_probs.save()
```

A list in one trace is left-padded, so `[:, -1]` is every reference prompt's last
token.

### Whole blocks: `skip_layers`

```python
with model.trace(prompt):
    model.skip_layers(LAYER, LAYER + 1)          # blocks LAYER..LAYER+1 do not run; the stream passes straight through
    probs = model.next_token_probs.save()
```

Negative indices count from the end. A skip must cover every row of a forward, so
this goes in a trace of its own rather than in one invoke of a batched trace (see
Gotchas).

### One head

The per-head outputs before the output projection are
`attention_head_outputs`, `[batch, seq, heads, head_dim]`; zero one head's slice:

```python
HEAD = 3
with model.trace(prompt):
    model.layers[LAYER].self_attn.attention_head_outputs[..., HEAD, :] = 0
    probs = model.next_token_probs.save()
```

Zeroing the head's *pattern* is the same ablation, and gives the same logits:

```python
with model.trace(prompt):
    model.layers[LAYER].self_attn.attention_probabilities[:, HEAD] = 0
    probs = model.next_token_probs.save()
```

Both need the eager attention path. Do not slice the projected `attention_output`
into `head_dim`-wide column blocks: after the output projection the hidden axis no
longer decomposes per head, and that edit removes something that is not the head.

### Every head of a block in one forward

One invoke per head, with the container made outside the trace:

```python
rows = []                                    # a name bound inside the trace does not survive it
with model.trace() as tracer:
    for head in range(model.num_heads):
        with tracer.invoke(prompt):
            model.layers[LAYER].self_attn.attention_head_outputs[..., head, :] = 0
            rows.append(model.next_token_probs[0, target].save())

print([round(float(p), 4) for p in rows])
```

Do this through `attention_head_outputs`, not the pattern. A value nnter reads inside a
nested `.source` call (`attention_probabilities`, `attention_scores`) cannot be touched in
two invokes of one trace today: the second invoke raises `TypeError: 'NoneType' object is
not subscriptable` (an nnsight bug). For per-head pattern edits, use one trace per head.

### A DeltaNet mixer on a hybrid

On Qwen3.5 three blocks in four carry `linear_attn` instead of `self_attn`. Its
contribution is `attention_output` too, so the block's mixer is one helper decided
outside the trace, and the ablation is the same line:

```python
model = StandardizedTransformer("Qwen/Qwen3.5-9B", dispatch=True, attn_implementation="eager")

def mixer(layer):
    attn = getattr(layer, "self_attn", None)
    return attn if attn is not None else layer.linear_attn

mixers = [mixer(layer) for layer in model.layers]      # outside: getattr on a served value inside a trace is fragile

outs = []
with model.trace() as tracer:
    with tracer.invoke(prompt):
        base = model.logits[:, -1].float().log_softmax(-1).save()
    for i in range(model.num_layers):
        with tracer.invoke(prompt):
            mixers[i].attention_output[:] = 0
            outs.append(model.logits[:, -1].float().log_softmax(-1).save())

for i, logprobs in enumerate(outs):
    print(f"block {i:2d} mixer KL {F.kl_div(logprobs, base, log_target=True, reduction='sum'):.4f}")
```

The same loop runs on a non-hybrid, where every `mixer(layer)` is `self_attn`; the
per-family version of it is in [cross-family-sweep](cross-family-sweep.md).

### Measuring on the logits instead

`next_token_probs` is `logits[:, -1].softmax(-1)`, derived and read-only. Read
`model.logits` for a logit difference (`logits[0, -1, paris] - logits[0, -1, rome]`)
or for a position other than the last.

## Interpretation

- Measure the functional effect: a probability, a logit gap, a KL, a task metric.
  Not the activation's norm.
- Zero ablation pushes the stream off-distribution; a large drop can mean
  "anything missing here breaks the model". Mean ablation asks the narrower
  question, whether the *deviation from average* matters.
- Two components that back each other up each ablate to nothing; ablate sets.
- Zero, mean and resample answer different questions; say which you ran.

## Gotchas

- `skip_layers` inside one invoke of a batched trace raises
  `ValueError: A batched .skip() has to cover every row`. Skip in every invoke or
  in a trace of its own.
- `[:] = 0` is in place. `.clone().save()` first if you also want the clean value.
- `next_token_probs` cannot be assigned; assign `logits`.
- `[:, -1]` is the last token of every row only under left padding.
- A checkpoint may lack the component: OPT has no `mlp` module, a hybrid's linear
  blocks have no `self_attn`. `model.support()` says so per block before any trace;
  reading anyway raises `nnter.Unavailable`.
- Decide `self_attn` versus `linear_attn` outside the trace, as above.
- Head-level ablation needs `attn_implementation="eager"`; the boundary values
  (`attention_output`, `mlp_output`, `layer_output`) do not.
- On Gemma-4 the KV-sharing blocks attend with an earlier block's keys and values.
  Zeroing `attention_keys` / `attention_values`, in place or by assignment, ablates
  them on that block alone; to ablate what every borrower attends with, edit the
  source block's `k_proj` / `v_proj` output
  ([Borrowed keys and values](../reference/families.md#borrowed-keys-and-values)).
- On Granite (and GraniteMoE, Granite-SWA, HyperCLOVA X, ZAYA) a write to
  `attention_output` or `mlp_output` changes the other positions by rounding;
  load in float32 for small edits ([families](../reference/families.md#scaled-residual-adds)).
- A name bound inside the trace does not survive it; make `rows`/`outs` outside.

## Related

- [activation-patching](activation-patching.md): paste a clean activation instead of
  zeroing.
- [attention-patterns](attention-patterns.md): the pattern the head edit touches.
- [contribution-decomposition](contribution-decomposition.md): what each component
  writes toward the answer, without removing it.
- [cross-family-sweep](cross-family-sweep.md): the mixer/MLP ablation over several
  checkpoints.
- [../usage/residual-stream.md](../usage/residual-stream.md): the contribution
  identity behind `attention_output` and `mlp_output`.
- [../usage/availability.md](../usage/availability.md): `support()`.
- nnsight `docs/usage/skip.md`, `docs/usage/invoke-and-batching.md`.
