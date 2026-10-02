---
title: Steering
one_liner: "Derive a direction from two contrasting prompts as a difference of `layer_output`, then `model.steer(layer, vector, factor, token_positions=-1)` adds it to the residual stream in a trace or at every step of a generate."
tags: [patterns, steering, residual-stream, generation]
related: [docs/usage/residual-stream.md, docs/usage/generation.md, docs/patterns/probing.md, docs/patterns/ablation.md, docs/patterns/activation-patching.md]
sources: [nnter/standardized.py, nnter/components/layer.py]
---

# Steering

## What this is for

Activation steering adds a fixed vector to the residual stream at one block and
lets every later block read the modified stream. The vector is usually a concept
direction: the difference of activations between a positive and a negative prompt
(or prompt set), or a probe's weight. Adding it pushes the model toward the
concept; subtracting suppresses it.

The stream leaving block `i` is `model.layers[i].layer_output` on every family, so
deriving the vector and applying it are the same two blocks of code everywhere;
`model.steer` is the in-place add written once. That is the normalization this
page leans on, and it is why it is shorter than a per-architecture recipe.

## Canonical pattern

Derive the direction from two prompts in one trace, two invokes, at the last
position:

```python
import torch
from nnter import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", dispatch=True)
LAYER = model.num_layers // 2

with model.trace() as tracer:
    with tracer.invoke("I love this so much"):
        positive = model.layers[LAYER].layer_output[:, -1].save()   # [1, hidden]
    with tracer.invoke("I hate this so much"):
        negative = model.layers[LAYER].layer_output[:, -1].save()

vector = (positive - negative)[0]
vector = vector / vector.norm()                                      # unit norm: factors compare across directions
```

Apply it in a new trace and compare with the unsteered run, again as two invokes so
both rows come from one forward. The factor is a fraction of the stream's norm at the
layer you steer ([Choosing the factor](#choosing-the-factor)):

```python
prompt = "I went to the bakery and"

with model.trace(prompt):
    scale = model.layers[LAYER].layer_output[0, -1].norm().save()   # the stream's norm where you steer
factor = 0.25 * float(scale)                                         # a fraction of it; sweep the fraction

with model.trace() as tracer:
    with tracer.invoke(prompt):
        baseline = model.next_token_probs.save()
    with tracer.invoke(prompt):
        model.steer(LAYER, vector, factor=factor, token_positions=-1)
        steered = model.next_token_probs.save()

print(model.probs_to_dict(baseline[0], k=3))     # {token: probability}
print(model.probs_to_dict(steered[0], k=3))
```

`steer(layers, vector, factor, token_positions, batch_index)` adds
`factor * vector` in place to `layer_output` at the given positions and rows (both
default to all), after moving the vector to the stream's device and dtype. The
add is in place on the served tensor, so it reaches the model; the invoke that
does not call it is untouched.

## By hand

`steer` is one line of the kind you would write yourself, and the two give the
same probabilities:

```python
with model.trace(prompt):
    out = model.layers[LAYER].layer_output
    out[:, -1] += factor * vector.to(out)
    by_hand = model.next_token_probs.save()
```

Use the hand form when the edit is not an add: projecting a direction out
(`out[:, -1] -= (out[:, -1] @ vector)[:, None] * vector`), clamping, or adding a
different vector per position.

## Choosing the factor

A raw factor means nothing across models or layers: the stream's norm differs by orders
of magnitude between checkpoints and changes with depth. On GPT-2 the stream leaving
block 6 has norm 92 at the last position, so `factor=4.0` with a unit vector leaves the
greedy continuation of the canonical prompt unchanged; on gemma-3-270m the same block's
norm is 7264. On Gemma-4 the norm also rises and falls with depth (24 to 92 across
gemma-4-E2B's blocks), so a factor measured at one layer is wrong at another. Measure the
norm at the layer you steer, as the canonical pattern does, and sweep the *fraction*.

Where the band lies is a property of the model, not a constant. With the canonical
direction and prompt, steered at every step of a 15-token greedy generation: GPT-2 changes
its continuation at 0.25 of the norm, stays fluent at 0.5 and repeats itself from 1.0;
gemma-3-270m drifts at 0.25 and repeats one token from 0.5. Start well below the norm
(0.05 to 0.1) and sweep layer and fraction together, reading generations, since the band
moves with the layer.

## Variations

### A direction from prompt sets

Replace the two prompts with two lists; each invoke's `[:, -1]` is then a batch
of last positions, averaged over the set:

```python
positive_set = ["I love this so much", "This is wonderful", "I am very happy"]
negative_set = ["I hate this so much", "This is terrible", "I am very sad"]

with model.trace() as tracer:
    with tracer.invoke(positive_set):
        positive = model.layers[LAYER].layer_output[:, -1].mean(0).save()
    with tracer.invoke(negative_set):
        negative = model.layers[LAYER].layer_output[:, -1].mean(0).save()
```

A list in one invoke is left-padded, so `[:, -1]` is every prompt's last token.

### Several layers, all positions, one row

```python
with model.trace(prompt):
    model.steer([LAYER - 1, LAYER, LAYER + 1], vector, factor=factor / 3)   # ascending; all positions

with model.trace([prompt, prompt]):
    model.steer(LAYER, vector, factor=factor, token_positions=-1, batch_index=1)   # row 1 only
    both = model.next_token_probs.save()                                          # both[0] is the baseline
```

### Under `generate`, at every step

A bare `steer` in a `generate` body fires once, on the prefill, and not on the
decode steps; the output still changes because the prompt was read differently,
which makes the mistake look like a modest success. Put the call in a bounded
`tracer.iter`, and hold the run to the bound with `min_new_tokens`:

```python
N = 20
with model.generate(prompt, max_new_tokens=N, min_new_tokens=N, do_sample=False) as tracer:
    for step in tracer.iter[:N]:                                   # the prefill and every decode step
        model.steer(LAYER, vector, factor=factor, token_positions=-1)
    ids = tracer.result.save()

print(model.tokenizer.decode(ids[0]))
```

`token_positions=-1` is the right index across steps: on the prefill it is the
prompt's last token, and on each decode step the one token that step processes.
An absolute position is out of range from step 1 on. See nnsight
`docs/usage/iter-all-next.md` for the bound and `docs/usage/generate.md` for
`tracer.result`.

### Suppressing

`factor < 0` subtracts the direction; projecting it out (`hs - (hs @ v) v`) removes
only its component and is the form refusal-direction work uses.

## Interpretation

- Look at a generation, not only the top token: a direction can flip the argmax
  and wreck the continuation.
- Compare against a random direction of the same norm at the same layer; a
  concept direction should beat it.
- Steering at block `i` is cumulative: every later block reads the change. To ask
  whether block `i` *itself* carries the concept, use
  [activation-patching](activation-patching.md).

## Gotchas

- `steer` and `+=` edit the stream in place. `.clone()` first if you also want the
  unsteered tensor.
- `layers` must be ascending in one `steer` call, and a later trace line must not
  reach back before it: reads and writes follow the forward.
- The vector's device and dtype are handled (`vector.to(out)`), its shape is not:
  it must be `[hidden]` or broadcast to the selected slice.
- A name bound inside a trace or a generate body does not survive it; `vector` is
  computed from saved tensors outside, as above.
- On a hybrid (Qwen3.5), `layer_output` exists on every block, so steering a
  DeltaNet block is the same call as steering an attention block.
- `next_token_probs` reads the last position, which is every row's last token
  only under left padding; a list in one invoke is padded that way.

## Related

- [probing](probing.md): a probe's weight as the direction, and the random-direction
  control.
- [ablation](ablation.md), [activation-patching](activation-patching.md).
- [../usage/residual-stream.md](../usage/residual-stream.md): `layer_output` and
  the contribution values.
- [../usage/generation.md](../usage/generation.md): `generate` and per-step
  interventions.
- nnsight `docs/usage/iter-all-next.md`, `docs/usage/generate.md`.
- Turner et al. (2023), "Activation Addition"; Arditi et al. (2024), "Refusal in
  Language Models Is Mediated by a Single Direction".
