---
title: Cross-Family Sweep
one_liner: "Run one experiment over several checkpoints: loop over repo ids, `StandardizedTransformer(repo)`, guard on `model.support()`, decide `self_attn` versus `linear_attn` outside the trace, and collect a per-family table."
tags: [patterns, sweep, families, hybrids, availability]
related: [docs/usage/loading.md, docs/usage/availability.md, docs/usage/vocabulary.md, docs/patterns/ablation.md, docs/patterns/attention-patterns.md]
sources: [nnterp/standardized.py, nnterp/families/__init__.py, nnterp/components/standard.py, nnterp/families/opt.py, nnterp/families/qwen3_5_text.py]
---

# Cross-Family Sweep

## What this is for

A claim about "transformers" needs more than one checkpoint. The obstacle is that
each family spells its modules differently, returns tensors or tuples, has or lacks
a component, and a hybrid has two kinds of block. nnterp removes the spelling: one
trace body reads `layer_output`, `attention_output`, `mlp_output`,
`attention_probabilities` and `next_token_probs` on every family. What remains is
what genuinely differs between checkpoints, and this page is the loop that handles
those three things: a value a checkpoint lacks, a block that has `linear_attn`
instead of `self_attn`, and memory.

## Canonical pattern

The experiment: zero each block's mixer contribution and each block's MLP
contribution in turn, and record how far the next-token distribution moves (KL from
the clean run), one forward per checkpoint. Then per-layer residual norms and
attention entropy.

```python
import torch
import torch.nn.functional as F
from nnterp import StandardizedTransformer

REPOS = {
    "gpt2": "openai-community/gpt2",
    "llama": "meta-llama/Llama-3.1-8B",
    "qwen3_5": "Qwen/Qwen3.5-9B",            # a hybrid: three DeltaNet blocks in four
    "opt": "facebook/opt-125m",              # no MLP module
}
prompt = "The Eiffel Tower is in the city of"


def mixer(layer):
    """The block's sequence mixer: `self_attn`, or `linear_attn` on a hybrid's linear block."""
    attn = getattr(layer, "self_attn", None)
    return attn if attn is not None else layer.linear_attn


table = {}
for name, repo in REPOS.items():
    model = StandardizedTransformer(repo, dispatch=True, attn_implementation="eager")
    support = model.support()                                   # before any trace: None means available on every block
    has_mlp = support.get("mlp.mlp_output", "absent") is None   # .get: a module no block has (OPT) has no key
    mixers = [mixer(layer) for layer in model.layers]         # decided outside the trace
    attention_blocks = [i for i, layer in enumerate(model.layers) if getattr(layer, "self_attn", None) is not None]

    outs = {}                                                 # containers made outside the trace
    with model.trace() as tracer:
        with tracer.invoke(prompt):
            base = model.logits[:, -1].float().log_softmax(-1).save()     # log-probabilities
        for i in range(model.num_layers):
            with tracer.invoke(prompt):
                mixers[i].attention_output[:] = 0
                outs["mixer", i] = model.logits[:, -1].float().log_softmax(-1).save()
            if has_mlp:
                with tracer.invoke(prompt):
                    model.layers[i].mlp.mlp_output[:] = 0
                    outs["mlp", i] = model.logits[:, -1].float().log_softmax(-1).save()

    norms, entropy = {}, {}
    with model.trace(prompt):
        for i, layer in enumerate(model.layers):
            if i in attention_blocks:
                p = layer.self_attn.attention_probabilities
                entropy[i] = (-p * (p + 1e-12).log()).sum(-1).mean().save()
            norms[i] = layer.layer_output[0, -1].norm().save()

    kl = lambda logprobs: float(F.kl_div(logprobs, base, log_target=True, reduction="sum"))   # KL(clean || ablated)
    table[name] = {
        "kind": ["linear" if getattr(layer, "linear_attn", None) is not None else "attn" for layer in model.layers],
        "mixer_kl": [kl(outs["mixer", i]) for i in range(model.num_layers)],
        "mlp_kl": [kl(outs["mlp", i]) for i in range(model.num_layers)] if has_mlp else None,
        "resid_norm": [float(norms[i]) for i in range(model.num_layers)],
        "entropy": {i: float(entropy[i]) for i in attention_blocks},
    }
    del model
    torch.cuda.empty_cache()

for name, row in table.items():
    print(name)
    for key, value in row.items():
        print(f"  {key:11s}", value)
```

The trace bodies are identical for all four; the loop body differs from a
single-model script in exactly three lines: the `support()` read, the `has_mlp`
guard, and `mixers` decided outside the trace. On the hybrid the `kind` row reads
`['linear', 'linear', 'linear', 'attn', ...]`, `entropy` has one entry per attention
block, and `mixer_kl` has one per block because a DeltaNet mixer's contribution is
`attention_output` too. On OPT `mlp_kl` is `None`: no block has an MLP module, so
`support()` has no `mlp.mlp_output` key, which is why the guard uses `.get`.

## The three guards

**Availability.** `model.support()` returns, for every standard value, `None` when
every block has it or `{block: reason}` where some do not. Guard on it rather than
on `hasattr`, which raises `nnterp.Unavailable` for an unavailable value. A module no
block has (OPT's `mlp`) has no key, so read it with `.get`. The common reasons in a
sweep: `no self_attn module` (a hybrid's linear blocks), `runs 'sdpa'; load with attn_implementation='eager'` (the interior on a
non-eager load), and GPT-2's `reorder_and_upcast_attn`.

**Hybrids.** A block has either `self_attn` or `linear_attn`. Decide which outside
the trace: `getattr(envoy, name, None)` inside a trace can trip on served values,
and the list comprehension above is the safe form. `layer_output`, `mlp_output`
and the mixer's `attention_output` exist on both kinds of block, so most
experiments do not branch at all.

**Memory.** Load one checkpoint at a time; `del model` and `torch.cuda.empty_cache()`
between them. For 8B-class checkpoints pass `dtype=torch.bfloat16,
device_map="auto"` to `StandardizedTransformer` like any `TransformersModel`.

## Variations

### A per-family metric on the same prompt set

Replace the single prompt with a list in each invoke; the saved log-probabilities are then
`[prompts, vocab]` and the KL a vector. A list is left-padded, so the last position
is every prompt's last token.

### Layouts hold across the table

Every value has one layout on every family (`attention_probabilities` is
`[batch, heads, query, key]`, `attention_head_outputs` is
`[batch, seq, heads, head_dim]`), so a metric written once indexes correctly
everywhere. Sizes differ: read `model.num_heads`, `model.head_dim`,
`model.hidden_size` off the model, not off a config key, since the key differs by
family (`n_head`, `num_attention_heads`).

### The same sweep with another experiment

Swap the trace body for any recipe on this site: the [logit-lens](logit-lens.md)
top-1 grid, an [activation-patching](activation-patching.md) layer sweep, a
[probing](probing.md) curve. The loop, the guards and the table stay.

## Gotchas

- Take the KL on log-probabilities, as above. `next_token_probs` underflows to exact
  zeros (thousands per row on Pythia's float16 checkpoint), and `p * (p.log() - q.log())`
  is then NaN.
- The KL and norm values from different checkpoints are not directly comparable:
  vocabularies, depths and residual scales differ. Compare shapes of curves, or
  normalize per model.
- The last position is the same token only if every checkpoint tokenizes the prompt
  to end at the same word; check `model.tokenizer(prompt).input_ids` when a
  position matters.
- `attention_probabilities` needs `attn_implementation="eager"` on every load; a
  checkpoint's default is `sdpa`.
- A DeltaNet mixer under `flash-linear-attention` or `causal-conv1d` has no Python
  source to read; `support()` reports its interior values unavailable with that
  reason. Its `attention_output` is a module boundary and stays available.
- Names bound inside a trace do not survive it; every container above is made
  outside and every entry is a `.save()`.
- A `model_type` nnterp has no family for loads with the best-effort default family and a
  warning; check `model.support()` before sweeping it (or raises `UnsupportedFamily` when the
  default cannot standardize it).

## Related

- [ablation](ablation.md): the mixer/MLP ablation on one model.
- [attention-patterns](attention-patterns.md): the entropy metric.
- [../usage/loading.md](../usage/loading.md): load arguments, `dtype`, `device_map`.
- [../usage/availability.md](../usage/availability.md): `support()` and every
  reason string.
- [../usage/vocabulary.md](../usage/vocabulary.md): which native module each
  standard name reaches, per family.
