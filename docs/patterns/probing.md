---
title: Probing
one_liner: "`get_token_activations(model, prompts, layers, idx=-1)` gives `[layers, prompts, hidden]` at one position of every prompt under left padding; fit a closed-form linear probe per layer in torch and read the accuracy curve."
tags: [patterns, probing, activations, residual-stream]
related: [docs/usage/activations.md, docs/usage/prompt-utils.md, docs/patterns/steering.md, docs/patterns/logit-lens.md, docs/patterns/ablation.md]
sources: [nnterp/nnsight_utils.py, nnterp/standardized.py, nnterp/components/layer.py]
---

# Probing

## What this is for

A probe asks whether a property is linearly decodable from the residual stream:
collect the activation at one position of every prompt in a labelled set, fit a
linear classifier per layer, and report accuracy against depth. The curve says
where the property becomes readable.

Collection is the family-specific part, and nnterp's
`get_token_activations` does it against the standard values: one forward over the
whole prompt set, `model.layers[i].layer_output` at one position, on any family.
The probe itself is a few lines of torch. That is the normalization this page
uses; the controls are what make the result mean something.

## Canonical pattern

Build the dataset from one batched forward. The last position of every prompt
needs left padding, which nnsight sets at load for a causal model (with the EOS as pad
token where the tokenizer has none):

```python
import torch
from nnterp import StandardizedTransformer
from nnterp.nnsight_utils import get_token_activations

model = StandardizedTransformer("openai-community/gpt2", dispatch=True)
assert model.tokenizer.padding_side == "left"

positive = ["wonderful", "fantastic", "delightful", "excellent", "brilliant", "joyful", "superb", "lovely"]
negative = ["terrible", "awful", "dreadful", "disgusting", "horrible", "miserable", "dismal", "boring"]
templates = ["The movie was {}.", "I found the book {}.", "That meal was {}.", "Their performance was {}."]

texts = [t.format(w) for w in positive for t in templates] + [t.format(w) for w in negative for t in templates]
labels = torch.tensor([1.0] * (len(positive) * len(templates)) + [0.0] * (len(negative) * len(templates)))

with torch.no_grad():
    acts = get_token_activations(model, texts, idx=-1)          # [layers, prompts, hidden], on the CPU

assert acts.shape == (model.num_layers, len(texts), model.hidden_size)
```

`get_token_activations(model, prompts, layers=None, get_activations=None, idx=-1)`
reads every block by default (`layers` picks some), `layer_output` by default
(`get_activations=lambda m, i: m.layers[i].mlp.mlp_output` reads another site),
and raises before running if `idx` is negative and the tokenizer does not pad left.
For a set too large for one batch, `collect_token_activations_batched(model, texts,
batch_size)` chunks it and returns the same tensor.

Then one closed-form ridge probe per layer, standardized on the training rows only:

```python
generator = torch.Generator().manual_seed(0)
order = torch.randperm(len(texts), generator=generator)
split = int(0.7 * len(texts))
train, test = order[:split], order[split:]


def fit_probe(x, y, ridge=1e-2):
    """Ridge regression of the {-1, +1} label on [n, hidden]; returns a predict(x) -> bool tensor."""
    mean, std = x.mean(0), x.std(0) + 1e-6
    design = lambda z: torch.cat([(z - mean) / std, torch.ones(len(z), 1)], dim=1)
    z = design(x)
    penalty = torch.eye(z.shape[1]); penalty[-1, -1] = 0                 # no penalty on the bias
    w = torch.linalg.solve(z.T @ z + ridge * len(z) * penalty, z.T @ (2 * y - 1))
    return lambda x_new: design(x_new) @ w > 0


for layer in range(model.num_layers):
    x = acts[layer].float()
    predict = fit_probe(x[train], labels[train])
    accuracy = (predict(x[test]).float() == labels[test]).float().mean()
    print(f"layer {layer:2d}  test accuracy {float(accuracy):.2f}")
```

Output is a curve, one line per layer (values here are illustrative of the format,
not of any checkpoint):

```
layer  0  test accuracy 0.65
layer  1  test accuracy 0.70
 ...
```

Read its shape. High from layer 0 means the probe reads token identity ("wonderful"
and "terrible" are different tokens); a rise through the middle means something
the model computes; high only at the end may be the prediction itself.

## Controls

Each one kills a different false positive, for one more fit.

**Shuffled labels.** Anything above chance is memorization capacity, and means the
probe is too expressive for the dataset:

```python
shuffled = labels[torch.randperm(len(labels), generator=generator)]
predict = fit_probe(acts[2][train].float(), shuffled[train])
control = (predict(acts[2][test].float()).float() == shuffled[test]).float().mean()   # expect about 0.5
```

**Held-out template.** Train on three templates, test on the fourth; if accuracy
collapses, the probe learned the template.

```python
held_out = torch.tensor([i for i in range(len(texts)) if i % len(templates) == 3])
kept = torch.tensor([i for i in range(len(texts)) if i % len(templates) != 3])
predict = fit_probe(acts[2][kept].float(), labels[kept])
```

**Difference of means.** No fitting; the direction is the gap between class means.
More robust on small sets, and usually the more causally effective direction:

```python
x = acts[2].float()
direction = x[train][labels[train] == 1].mean(0) - x[train][labels[train] == 0].mean(0)
direction = direction / direction.norm()
threshold = (x[train] @ direction).mean()
accuracy = (((x[test] @ direction) > threshold).float() == labels[test]).float().mean()
```

## Causal check

A probe is correlational. To claim the model *uses* the direction, add it with
`model.steer` and check the behavior moves, and that a random direction of the same
norm does not:

```python
LAYER = 2
ids = [model.tokenizer(word, add_special_tokens=False).input_ids for word in (" great", " bad")]
assert all(len(i) == 1 for i in ids), [model.tokenizer.convert_ids_to_tokens(i) for i in ids]
good, bad = ids[0][0], ids[1][0]

def logit_gap(vector, factor):
    with model.trace("The movie was"):
        if vector is not None:
            model.steer(LAYER, vector, factor=factor, token_positions=-1)
        logits = model.logits[0, -1].save()
    return float(logits[good] - logits[bad])

with model.trace("The movie was"):
    scale = model.layers[LAYER].layer_output[0, -1].norm().save()
factor = 0.5 * float(scale)

baseline = logit_gap(None, 0)
probe = logit_gap(direction, factor)
negated = logit_gap(-direction, factor)
random = [logit_gap(torch.nn.functional.normalize(torch.randn_like(direction), dim=0), factor) for _ in range(8)]
```

`add_special_tokens=False` keeps the BOS token out of the ids (`tokenizer.encode(" great")[0]`
is the BOS id on Llama and Gemma); the assertion catches a word that is more than one token
(Mistral's sentencepiece tokenizer splits `" Paris"` into `['▁', '▁Paris']`), in which case try
it without the leading space or pick another word.

A direction that decodes and steers nothing is also a result: a feature the model
does not read. Report it rather than raising the factor until something moves; the
factor band is in [steering](steering.md).

## Variations

### Another site or position

`get_activations` names the site, `idx` the position. `idx=0` needs right padding
(`tokenizer_kwargs={"padding_side": "right"}`); a positive `idx` likewise. For a
property mentioned mid-prompt, the last position is usually the wrong one.

### A gradient-fit probe

When you want logistic loss rather than least squares, a few Adam steps on
`(weight, bias)` with `binary_cross_entropy_with_logits` and an L2 term replace
`fit_probe`; standardize on the training rows the same way.

### Target-token probabilities instead of activations

`nnterp.prompt_utils` (`Prompt.from_strings`, `run_prompts`) tracks named target
tokens in the next-token distribution over many prompts, the read-out side of a
behavioral dataset; see [prompt-utils](../usage/prompt-utils.md).

## Gotchas

- Wrap collection in `torch.no_grad()`: a trace runs with autograd on, and a saved
  activation otherwise pins the whole forward graph for every layer.
- A negative `idx` with a right-padding tokenizer raises
  `ValueError: a negative token index needs left padding`. nnsight left-pads a causal
  model by default, so this only fires after `tokenizer_kwargs={"padding_side": "right"}`
  or a tokenizer you set yourself.
- Fit on training rows only, including the standardization statistics.
- Keep the ridge (or weight decay) on and report it: with `hidden` far above the
  number of examples, an unregularized probe fits anything.
- Accuracy is not comparable across models of different `hidden_size` unless probe
  capacity is held fixed; report the curve, the example count and the layer-0 value.
- `get_token_activations` returns CPU tensors; move `direction` back with
  `model.steer`'s own `.to(out)` (it does) rather than by hand.

## Related

- [steering](steering.md): the factor band for the causal check.
- [logit-lens](logit-lens.md): the other read-only depth measurement.
- [ablation](ablation.md): does removing the site change the behavior.
- [../usage/activations.md](../usage/activations.md): `get_token_activations`,
  the batched and session variants, padding sides.
- [../usage/prompt-utils.md](../usage/prompt-utils.md): target tokens over prompt
  sets.
