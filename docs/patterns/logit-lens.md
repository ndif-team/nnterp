---
title: Logit Lens
one_liner: "`model.project_on_vocab(layer.layer_output)` reads each block's residual stream through the final norm, `lm_head` and any softcap, so one loop over `model.layers` is the lens on every family."
tags: [patterns, logit-lens, residual-stream, decoding]
related: [docs/usage/root-values.md, docs/usage/residual-stream.md, docs/patterns/contribution-decomposition.md, docs/patterns/activation-patching.md, docs/patterns/probing.md]
sources: [nnterp/standardized.py, nnterp/components/layer.py, nnterp/families/deepseek_v4.py]
---

# Logit Lens

## What this is for

The logit lens reads the residual stream leaving block `i` through the model's own
final norm and unembedding: what the model would predict if it stopped at block `i`.
Plotted over depth, it shows where an answer emerges and how the prediction is
refined.

In nnterp the three things the lens needs are standard on every family: the stream
is `model.layers[i].layer_output` (a tensor whether the block returns a tensor or a
tuple), and `model.project_on_vocab(hidden)` applies `model.norm`, `model.lm_head`
and the model's logit softcapping when its config has one. So the block below runs
unchanged on GPT-2, Llama, Gemma-2 or a Qwen3.5 hybrid; the standard values do the
per-architecture work, which is why this page is short.

## Canonical pattern

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", dispatch=True)
prompt = "The Eiffel Tower is in the city of"

lens = {}                                   # made outside the trace: a name bound inside does not survive it
with model.trace(prompt):
    for i, layer in enumerate(model.layers):
        lens[i] = model.project_on_vocab(layer.layer_output)[:, -1].save()   # [batch, vocab]
    logits = model.logits.save()

for i, row in lens.items():
    print(f"layer {i:2d}: {model.tokenizer.decode(row.argmax(-1)[0])!r}")

assert torch.allclose(lens[model.num_layers - 1], logits[:, -1])   # the last block's lens is the model's prediction
```

The loop reads the blocks in forward order, and `logits` after them, which is the
order one trace has to respect. The assertion is the wiring check: at the last
block the lens *is* the final computation, so it equals `model.logits`. It holds
on every family because `project_on_vocab` applies the same softcap the model
does (see [Softcapping](#softcapping)).

`get_topk_closest_tokens` turns a saved residual into `{token: probability}`:

```python
LAYER = model.num_layers // 2
with model.trace(prompt):
    resid = model.layers[LAYER].layer_output.save()          # [batch, seq, hidden]

model.get_topk_closest_tokens(resid[0, -1], k=5)             # one dict: the last position
model.get_topk_closest_tokens(resid[0], k=1)                 # one dict per position
```

It calls `project_on_vocab` then softmax, outside a trace on a saved tensor or
inside one on a live value.

## Softcapping

`model.logits` is the model's output; `model.lm_head.output` is the raw projection.
On Gemma-2 the two differ, because `Gemma2ForCausalLM.forward` applies
`cap * tanh(logits / cap)` after `lm_head` (`config.final_logit_softcapping`,
`30.0` on `google/gemma-2-2b`). `project_on_vocab` reads that config key and applies
the same cap, so intermediate blocks are read on the model's own scale, and the
wiring check above passes on Gemma-2 too. An uncapped lens on a softcapped model
is far too confident at every layer; the top-1 token usually survives, nothing
else does. Check `model.config.get_text_config()`, not a list of model names: Gemma-3
sets the key to `None`, Gemma-4 to `30.0`, and a multimodal wrapper (`gemma3`, `gemma4`)
keeps it in `text_config`, which is where `project_on_vocab` reads it. Gemma-4's
`layer_scalar` shrinks the stream between blocks; the final RMS norm divides the scale
back out, so the lens on an intermediate block reads it on the same footing as the last.

## Parallel streams (DeepSeek-V4)

DeepSeek-V4's `layer_output` is `[batch, seq, streams, hidden]`, several parallel copies
of the stream, and the model reads them out through `hc_head`, a learned weighting of the
streams, before `norm` and `lm_head`. The family's `project_on_vocab` does the same, so the
canonical pattern above runs unchanged there, its lens has the stream axis collapsed
(`[batch, seq, vocab]`), and the wiring check passes exactly. A lens on one stream is a
different readout, which the model never makes:

```python
model = StandardizedTransformer("deepseek-ai/DeepSeek-V4-Flash", dispatch=True)

with model.trace(prompt):
    resid = model.layers[3].layer_output.save()                   # [batch, seq, streams, hidden]

default = model.project_on_vocab(resid)                            # through hc_head: [batch, seq, vocab]
stream_0 = model.lm_head(model.norm(resid[:, :, 0]))                # stream 0 alone: [batch, seq, vocab]
```

## Variations

### Top-1 at every position

```python
with model.trace(prompt):
    grid = torch.stack([model.project_on_vocab(layer.layer_output)[0].argmax(-1)
                        for layer in model.layers]).save()   # [layers, seq]

tokens = [model.tokenizer.decode(t) for t in model.tokenizer(prompt).input_ids]
for i, row in enumerate(grid):
    print(f"layer {i:2d}:", [model.tokenizer.decode(t) for t in row])
```

Column `j` of the grid is the prediction *after* token `j`, so the label for column
`j` is `tokens[j]` and the value is a guess at `tokens[j + 1]`.

### Probability of a target token across layers

```python
ids = model.tokenizer(" Paris", add_special_tokens=False).input_ids
assert len(ids) == 1, model.tokenizer.convert_ids_to_tokens(ids)   # one token, or target is not the word
target = ids[0]

with model.trace(prompt):
    probs = torch.stack([model.project_on_vocab(layer.layer_output)[0, -1].softmax(-1)[target]
                         for layer in model.layers]).save()   # [layers]
```

`add_special_tokens=False` keeps the BOS token out of the target (`tokenizer.encode(" Paris")[0]`
is the BOS id on Llama and Gemma, and the curve is then flat at zero). The assertion
catches a word that is more than one token: Mistral's sentencepiece tokenizer gives
`['▁', '▁Paris']` and Granite's `['ĠPar', 'is']`. Then try the word without the leading
space (`"Paris"` is `['▁Paris']` on Mistral), or pick a target word that is one token.

Argmax hides a close race; the probability curve shows a peak that may come before
the final layer.

### Top-k per layer

```python
with model.trace(prompt):
    topk = torch.stack([model.project_on_vocab(layer.layer_output)[0, -1].topk(5).indices
                        for layer in model.layers]).save()    # [layers, 5]
```

### The lens on one sublayer's contribution

`project_on_vocab` takes any `[..., hidden]` tensor, so it also reads what one
sublayer *adds*: `model.project_on_vocab(layer.self_attn.attention_output)` or
`model.project_on_vocab(layer.mlp.mlp_output)`. The final norm is nonlinear, so
those do not sum to the block's lens; the linear form and the running-sum check
are in [contribution-decomposition](contribution-decomposition.md).

## Interpretation

- Read the curve, not one layer. The layer where the answer first becomes top-1 is
  where the bulk of the decision is made; later layers refine.
- A flat or late curve is not a failure of the model. Many checkpoints decode
  punctuation or frequent tokens for most of their depth and the answer only in
  the last few blocks, while answering correctly. Once the wiring check passes,
  that is a fact about the lens.
- `[:, -1]` reads the next-token prediction. For factual recall the subject token's
  position is often the informative one; use the grid.
- Keep the norm. Decoding `lm_head(hidden)` without `model.norm` is not a cheaper
  lens; a norm is not affine, and skipping it produces a confident, wrong curve
  rather than an error. `project_on_vocab` includes it, which is the point.

## Gotchas

- `project_on_vocab` *calls* `model.norm` and `model.lm_head` inside the trace. nnsight
  runs a module called this way stood down, as plain arithmetic; reading
  `model.lm_head.output` is a different thing, the model's own call. See nnsight
  `docs/usage/access-and-modify.md`.
- Reads follow the forward within one trace: the layer loop is in order by
  construction, and `model.logits` comes after it. Reading `logits` first and then a
  block raises `OutOfOrderError`.
- A name bound inside the trace body does not survive it unless it is a `.save()`;
  make the container (`lens = {}`) outside and save each entry.
- On a batch of prompts, `[:, -1]` is the last token of every row only under left
  padding; see [activations](../usage/activations.md).
- The lens does not need the eager attention path. It reads module boundaries,
  so it works on an `sdpa` load; only the attention interior needs eager.
- Reading every layer's `[batch, seq, vocab]` is `layers * seq * vocab` floats;
  index the position or take `argmax` inside the trace, as above.

## Related

- [contribution-decomposition](contribution-decomposition.md): the lens on each
  sublayer's contribution, and the linear form that sums.
- [activation-patching](activation-patching.md): once the lens says *where* the
  answer appears, patching says whether that layer carries it.
- [probing](probing.md): the other read-only depth measurement.
- [../usage/root-values.md](../usage/root-values.md): `logits` versus
  `lm_head.output`, `next_token_probs`.
- [../usage/residual-stream.md](../usage/residual-stream.md): `layer_output` and
  the contributions.
- nnsight `docs/usage/access-and-modify.md`: calling modules inside a trace.
- nostalgebraist (2020), "interpreting GPT: the logit lens".
