---
title: Usage Index
one_liner: One page per nnter feature, recipe-style; every snippet has run against the pinned checkpoints.
tags: [usage, index]
related: [docs/patterns/index.md, docs/extending/index.md, docs/reference/api-quick-reference.md]
sources: [nnter/standardized.py, nnter/components/__init__.py]
---

# Usage Index

nnter adds names and values to an nnsight `TransformersModel`; everything nnsight does
(`trace`, `generate`, `.save()`, invokes, `tracer.iter`, `.source`, `remote=True`) is unchanged
and documented in nnsight's own `docs/usage/`. These pages cover what nnter adds.

All examples start from:

```python
from nnter import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", dispatch=True, attn_implementation="eager")
```

## Loading and names

- [loading](loading.md) — `StandardizedTransformer(repo_id, ...)`: family from `model_type`, `rename=`/`envoys=` merge, `tokenizer_kwargs`, `attn_implementation="eager"`.
- [vocabulary](vocabulary.md) — `embed_tokens`, `layers[i].self_attn`, `layers[i].mlp`, `norm`, `lm_head` on every family, as aliases beside the native names; why norms are not in it.

## The standard values

- [residual-stream](residual-stream.md) — `layer_output`, `attention_output`, `mlp_output`, and the identity `input + attention_output + mlp_output == layer_output`.
- [attention-interior](attention-interior.md) — `attention_queries` / `keys` / `values` / `scores` / `probabilities` / `head_outputs` inside the eager attention forward, with each family's caveats.
- [root-values](root-values.md) — `logits`, `token_embeddings`, `next_token_probs`, `input_ids`, `attention_mask`, `input_size`, and the sizes.
- [delta-net](delta-net.md) — the hybrids' `linear_attn`: `decays`, `betas`, `state_input`/`state_output`, and the per-token `state`/`states` behind `route_kernels`.
- [selective-scan](selective-scan.md) — Mamba, Falcon-Mamba and Jamba's `linear_attn`: the Mamba-1 scan's `C`/`B`/`x`, step sizes and decays as tokens-first views, and the scan's own per-token state behind `route_kernels`.
- [state-space](state-space.md) — the Mamba-2 (SSD) `linear_attn` on Mamba-2, Nemotron-H, Bamba and Falcon-H1: `C`/`B`/`x`/`dt` under the shared names, the state between steps.
- [mixture-of-experts](mixture-of-experts.md) — a mixture's `router_logits`, `expert_weights` / `expert_indices` (`[batch, seq, top_k]`), `expert_outputs`, `routed_output`, `shared_expert_output` on `layers[i].mlp` (a `Moe`); ablation, rerouting, `experts_implementation=`.
- [layouts](layouts.md) — one axis layout per value on every family, a named `jaxtyping` type from `nnter.components` (`Residual`, `Pattern`, ...; `value.dims`, `value.layout`).
- [availability](availability.md) — `model.support()`, `nnter.Unavailable`, and the reasons a checkpoint lacks a value.

## Doing things with them

- [methods](methods.md) — `skip_layers`, `steer`, `project_on_vocab`, `get_topk_closest_tokens`, `probs_to_dict`.
- [generation](generation.md) — the values under `model.generate`: per forward call, `[batch, 1, ...]` on a decode step, `tracer.iter` picks the step.
- [prompt-utils](prompt-utils.md) — `nnter.prompt_utils`: target-token probability mass over many prompts.
- [activations](activations.md) — `nnter.nnsight_utils`: one position's activation at chosen blocks over many prompts.
- [remote](remote.md) — `remote=True`: what travels, what the server needs.

## Related

- [docs/patterns/index.md](../patterns/index.md) — recipes written against these values.
- [docs/reference/families.md](../reference/families.md) — every family's quirks in one table.
