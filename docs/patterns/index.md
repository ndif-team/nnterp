---
title: Patterns Index
one_liner: Interpretability recipes written once against nnterp's standard values, so each runs unchanged on every family.
tags: [pattern, interpretability, index]
related: [docs/usage/index.md, docs/reference/families.md]
sources: [nnterp/standardized.py, nnterp/components/layer.py, nnterp/components/attention.py, nnterp/components/vision.py]
---

# Patterns Index

One technique per page: the smallest working example, then variations, then gotchas. The
recipes are shorter than nnsight's own because the standard values do the normalization: a
block's residual stream is `layer_output` on every family, a sublayer's contribution is
`attention_output` / `mlp_output`, the pattern is `attention_probabilities`, and
`project_on_vocab` is the model's own norm, unembedding and softcap. nnsight's
`docs/patterns/` has the same techniques written against raw module paths.

Every example loads eager, so the attention interior is available:

```python
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", dispatch=True, attn_implementation="eager")
```

## Look at activations

- [logit-lens](logit-lens.md) — `project_on_vocab(layer.layer_output)` per block; where the answer emerges.
- [attention-patterns](attention-patterns.md) — `attention_probabilities` as `[batch, heads, query, key]`; head metrics; editing the pattern.
- [contribution-decomposition](contribution-decomposition.md) — direct logit attribution over `attention_output` / `mlp_output`, and per head via `attention_head_outputs`.

## Modify activations

- [ablation](ablation.md) — zero or mean ablate a contribution, a head, a DeltaNet mixer, or whole blocks with `skip_layers`.
- [expert-ablation](expert-ablation.md) — every routed expert's effect on a target token, by zeroing `expert_weights` where `expert_indices == e`, on every MoE family.
- [activation-patching](activation-patching.md) — patch `layer_output` or a contribution from a clean run into a corrupt one; sweep layers and positions.
- [steering](steering.md) — a direction from contrasting prompts, added with `model.steer` in a trace or at every step of a generate.

## Across models and datasets

- [cross-family-sweep](cross-family-sweep.md) — one experiment over several checkpoints, guarded by `support()`, hybrids handled.
- [probing](probing.md) — `get_token_activations` to build a dataset, a closed-form linear probe per layer.

## Vision-language models

- [image-pathway](image-pathway.md) — ablate the image at `vision.image_features` or inside the tower, patch one image's features into another's run, each head's attention onto the image, and one-sided edits at the image positions of a text block.

## Hybrids

- [delta-net-state](delta-net-state.md) — read, patch and track the gated DeltaNet recurrent state on Qwen3-Next / Qwen3.5.

## Related

- [docs/usage/index.md](../usage/index.md) — the values these recipes are written against.
- [docs/usage/availability.md](../usage/availability.md) — guard a recipe on what the checkpoint has.
