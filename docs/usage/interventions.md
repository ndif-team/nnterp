---
title: Interventions
one_liner: "`nnterp.interventions`: `logit_lens`, `patchscope_lens`, `patchscope_generate` and `patch_object_attn_lens`, each a whole experiment in one call that returns `[num_prompts, num_layers, vocab]` (or generated ids) on the CPU, on every family."
tags: [usage, interventions, logit-lens, patchscope, TargetPrompt, repeat_prompt]
related: [docs/patterns/logit-lens.md, docs/patterns/activation-patching.md, docs/usage/methods.md, docs/usage/activations.md, docs/usage/availability.md]
sources: [nnterp/interventions.py, nnterp/nnsight_utils.py, nnterp/standardized.py]
---

# Interventions

## What this is for

Four lenses that open their own traces and hand back tensors, for when you want the
result rather than the trace: the logit lens at the last token of every prompt, a
patchscope (a hidden state from one prompt written into a *target prompt*, read as a
distribution or as a generation), and the object-attention lens. They are written
against the standard values (`layer_output`, `self_attn.input`, `next_token_probs`,
`project_on_vocab`), so a family's own arithmetic stays the family's and every lens runs
on every family. All are importable from `nnterp` and from `nnterp.interventions`.

| function | returns |
| --- | --- |
| `logit_lens(nn_model, prompts, remote=False, return_inv_logits=False)` | `[num_prompts, num_layers, vocab]` probabilities; with `return_inv_logits`, also the lens on `-hidden` |
| `patchscope_lens(nn_model, source_prompts=None, target_patch_prompts=None, layers=None, latents=None, remote=False)` | `[num_sources, len(layers), vocab]`: the target's next-token distribution with block `l`'s stream patched |
| `patchscope_generate(nn_model, prompts, target_patch_prompt, max_length=50, layers=None, remote=False, max_batch_size=32)` | `{layer: ids}`, `[num_prompts, seq]` generated from the patched target |
| `patch_object_attn_lens(nn_model, source_prompts, target_prompts, attn_idx_patch, num_patches=5)` | `[num_targets, num_layers, vocab]`: the attention inputs at one target position replaced over a window of blocks |

## Canonical pattern

```python
from nnterp import StandardizedTransformer, logit_lens, patchscope_lens, repeat_prompt

model = StandardizedTransformer("gpt2")
prompts = ["The Eiffel Tower is in the city of", "The capital of Japan is"]

probs = logit_lens(model, prompts)                 # [2, 12, 50257]: the last token, read off every block
model.tokenizer.decode(probs[0, -1].argmax())      # the model's own prediction: the last block's row is next_token_probs

target = repeat_prompt()                           # "king king\n1135 1135\nhello hello\n?", patched at the "?"
scope = patchscope_lens(model, prompts, target, layers=[2, 6, 11])   # [2, 3, 50257]
model.tokenizer.decode(scope[0, -1].argmax())      # what block 11's stream at "of" decodes to in the target
```

## Target prompts

`TargetPrompt(prompt, index_to_patch)` is a target and the token position written into.
`repeat_prompt(words=None, rel=" ", sep="\n", placeholder="?", index_to_patch=-1)` builds the
patchscopes paper's next-token target, and `it_repeat_prompt(tokenizer, ...)` the same inside the
tokenizer's chat template for instruction-tuned checkpoints. `patchscope_lens` takes one
`TargetPrompt` (used for every source), a list of them (one per source) or a
`TargetPromptBatch`; `TargetPromptBatch.from_prompts(prompts, index_to_patch)` builds one from
strings. Instead of `source_prompts`, `latents` takes hidden states already collected, in the
layout `nnsight_utils.get_token_activations` returns: `[len(layers), num_sources, hidden]`.

## Families

- **Parallel streams (DeepSeek-V4).** `layer_output` is `Streams`, `[batch, seq, streams, hidden]`.
  `logit_lens` reads each block through `project_on_vocab`, which collapses the streams the way
  the model's readout does (`hc_head`), so it returns `[num_prompts, num_layers, vocab]` there too
  and its last row is `next_token_probs`. A patchscope writes all of a position's streams, and its
  `latents` carry the stream axis (`[len(layers), num_sources, streams, hidden]`).
- **No softmax attention on a block.** `patch_object_attn_lens` patches every block's
  `self_attn.input`, so on a hybrid (a block with `linear_attn` only) or a state-space model it
  raises `nnterp.Unavailable`, naming the blocks without one.
- **Padding.** The lenses read the last position, which is every prompt's last token only under
  left padding (nnsight's default); `logit_lens` raises `ValueError` on a right-padding tokenizer.
  A tokenizer without a pad token cannot batch a list of prompts: pass
  `tokenizer_kwargs={"pad_token": ...}` at load.

## Related

- [logit-lens](../patterns/logit-lens.md): the lens written by hand, every position, a target token's curve.
- [activation-patching](../patterns/activation-patching.md): patching with a metric, by hand.
- [activations](activations.md): `get_token_activations`, which collects `latents`.
- [methods](methods.md): `skip_layer`, `skip_layers`, `steer`, `project_on_vocab`.
