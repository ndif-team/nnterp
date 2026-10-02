---
title: Expert Ablation
one_liner: "Every routed expert's effect on a target token, by zeroing `expert_weights` where `expert_indices == e`: one trace per expert, or one invoke per expert with an in-place edit; the same code on every MoE family."
tags: [patterns, moe, mixture-of-experts, ablation, experts]
related: [docs/usage/mixture-of-experts.md, docs/patterns/ablation.md, docs/usage/availability.md]
sources: [nnterp/components/moe.py, tests/families/suite.py]
---

# Expert Ablation

## What this is for

Removing one expert from a mixture asks whether a prediction depends on it. The
removal is a write of the routing the model consumes: zero the weight of every slot that
chose expert `e`, and those slots add nothing ([mixture-of-experts](../usage/mixture-of-experts.md)).
The other slots keep their weights, so the token is not renormalized onto its remaining
experts; that is the ablation, not a re-run of the router.

## Canonical pattern

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("hf-internal-testing/tiny-random-MixtralForCausalLM", dispatch=True)
prompt = "The Eiffel Tower is in the city of"
ids = model.tokenizer(" Paris", add_special_tokens=False).input_ids
if len(ids) > 1:                                                   # sentencepiece: ['▁', '▁Paris']
    ids = model.tokenizer("Paris", add_special_tokens=False).input_ids   # ['▁Paris']: it marks the word start itself
assert len(ids) == 1, model.tokenizer.convert_ids_to_tokens(ids)   # one token, or target is not the word
target = ids[0]
blocks = [layer.mlp for i, layer in enumerate(model.layers) if model.support(layer=i).get("mlp.expert_weights", "absent") is None]

with model.trace(prompt):
    clean = model.logits[0, -1].float().log_softmax(-1)[target].save()

effects = torch.zeros(len(blocks), blocks[0].num_experts)      # [block, expert]: change in log p(target)
for b, moe in enumerate(blocks):
    for e in range(moe.num_experts):
        with model.trace(prompt):
            moe.expert_weights = moe.expert_weights.masked_fill(moe.expert_indices == e, 0)
            effects[b, e] = (model.logits[0, -1].float().log_softmax(-1)[target] - clean).save()
```

`support()` picks the blocks with the routing pair: dense blocks beside mixture blocks
(DeepSeek-V3, GLM-4-MoE, Llama 4, Jamba) have none, and Llama 4's `expert_weights` is
unavailable. `add_special_tokens=False` keeps the BOS token out of the target. This
checkpoint's sentencepiece tokenizer (Mistral's) splits `" Paris"` into `['▁', '▁Paris']`,
so the recipe falls back to the word without its space; a tokenizer with no single token
for the word fails the assertion (`tokenizer(" Paris").input_ids[-1]` would silently be
`'is'` on Granite, which splits it into `['ĠPar', 'is']`), and then another word is the
target to pick.

Compare log-probabilities in float32, and for a sweep over single experts load the model
in float32 too. One expert's effect is small (a few hundredths of a nat on
granite-3.0-1b-a400m), and in bf16 the rounding of the rest of the forward is the same
size: on that checkpoint on a GPU the clean run alone and the same prompt as one invoke
of a batch already differ by 0.003 in log p, comparable to most experts' effects, and the
bf16 effects differ from the float32 ones by as much as the effects themselves. In float32 the per-trace and
per-invoke sweeps below agree to about 1e-5.

## One invoke per expert

The sweep of one block fits in one trace, one invoke per expert, with the clean baseline
as an unedited invoke of the same batch. Under several invokes, edit in place: each
invoke is served its own rows of the flat routing tensors, and an in-place edit reaches
that invoke alone, to rounding.

```python
moe = blocks[1]
rows = []  # bound outside: a name bound inside the block does not survive it
with model.trace() as tracer:
    with tracer.invoke(prompt):                    # unedited: the baseline, rounded the way the edited rows are
        base = model.logits[0, -1].float().log_softmax(-1)[target].save()
    for e in range(moe.num_experts):
        with tracer.invoke(prompt):
            moe.expert_weights[:] = moe.expert_weights.masked_fill(moe.expert_indices == e, 0)
            rows.append(model.logits[0, -1].float().log_softmax(-1)[target].save())
torch.stack([row - base for row in rows])         # == effects[1] in float32
```

The baseline comes from the batch, not from the single-prompt `clean` above, because a
batch is computed with other kernels and rounding than one prompt: in bf16 that difference
alone can exceed one expert's effect.

An assignment (`moe.expert_weights = ...`) under two or more invokes needs nnsight's
widen of an edit to a tensor whose leading axis is not the batch (nnsight PR #738).

## Gotchas

- **Unused experts read zero.** An expert no token of the prompt chose has no slots to
  zero; its effect is exactly zero, not "unimportant".
- **ZAYA's skip slots alias expert 0.** A skipped slot has index 0 and weight 0; zeroing
  `expert_indices == 0` zeroes it again, harmlessly, but count usage on `w != 0`.
- **The shared expert is not an expert here.** Ablate it with `shared_expert_output[:] = 0`.
- **Zero ablation overstates.** As with any zero ablation, the block adds less than
  it ever does; rerouting the slots to another expert (`expert_indices`) is the milder
  intervention.
