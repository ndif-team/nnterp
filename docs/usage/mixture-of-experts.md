---
title: Mixture of Experts
one_liner: Read, ablate and reroute a mixture of experts on every MoE family — `router_logits`, `expert_weights`, `expert_indices`, `expert_outputs`, `routed_output`, `shared_expert_output` on `layers[i].mlp` (a `Moe`), the sparse `[batch, seq, top_k]` routing pair, `experts_implementation=` and the per-family caveats.
tags: [usage, moe, mixture-of-experts, router, experts, routing, ablation, top_k]
related: [docs/usage/residual-stream.md, docs/usage/layouts.md, docs/usage/availability.md, docs/patterns/expert-ablation.md, docs/reference/families.md]
sources: [nnterp/components/moe.py, nnterp/components/eproperty.py, nnterp/families/mixtral.py, nnterp/families/deepseek_v3.py, nnterp/families/gemma4_text.py, nnterp/families/llama4_text.py, nnterp/families/jetmoe.py, nnterp/families/dbrx.py, nnterp/families/zaya.py, nnterp/families/nemotron_h.py, nnterp/families/deepseek_v4.py, tests/families/suite.py]
---

# Mixture of Experts

## What this is for

On a mixture-of-experts block, `mlp_output` is still what the block adds to the residual
stream. The six values below are how the mixture made it: the router's logits, the
experts and weights it chose for each token, each chosen expert's weighted output, the
routed sum, and the shared expert beside it. Expert ablation, rerouting, expert usage and
routing entropy are reads and writes of these. They live on `model.layers[i].mlp` when it
is a `Moe` (an `Mlp` subclass), on all 39 MoE families, with one layout each.

Nearly every mixture in transformers computes the same thing:

```
logits, w, idx = router(x)          # w, idx: [tokens, top_k]; tokens = batch * seq
routed = experts(x, idx, w)         # [tokens, hidden]
out    = routed (+ shared(x))
```

so nnterp reads each value where the model consumes it: the logits inside the router's
forward before the scoring, the weights and indices as the experts module's arguments,
the per-slot outputs inside transformers' grouped experts forward, the routed sum as the
experts' output. Every write lands on the tensor the model uses.

## Canonical pattern

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("hf-internal-testing/tiny-random-MixtralForCausalLM")
moe = model.layers[1].mlp
moe.num_experts, moe.top_k, moe.SCORING     # (4, 2, 'softmax')

with model.trace("The Eiffel Tower is in the city of"):
    logits = moe.router_logits.save()       # [batch, seq, experts]      before the softmax
    w = moe.expert_weights.save()           # [batch, seq, top_k]        what scales each slot
    idx = moe.expert_indices.save()         # [batch, seq, top_k]        int64, the chosen experts
    slots = moe.expert_outputs.save()       # [batch, seq, top_k, hidden]
    routed = moe.routed_output.save()       # [batch, seq, hidden]       == slots.sum(2)
    out = moe.mlp_output.save()             # what the block adds; == routed here (no shared expert)
```

The reads are in forward order, so they fit in one trace. `slots.sum(2) == routed` to the
dtype's rounding, and `routed_output + shared_expert_output` is the mixture's output.

## The six values

| value | what it is | layout | where it is read |
| --- | --- | --- | --- |
| `router_logits` | the router's logits, before the scoring and any selection bias | `RouterLogits`: `batch seq experts` | the op in the router's forward that produces them (`F_linear_0`, or the router's projection module) |
| `expert_weights` | the weight each routing slot's expert output is scaled by, after the router's normalization and scaling | `ExpertWeights`: `batch seq top_k` | the experts module's third argument |
| `expert_indices` | the expert each slot sends the token to | `ExpertIndices`: `batch seq top_k`, int64 | the experts module's second argument |
| `expert_outputs` | each slot's expert output times its weight, before the slots are summed | `ExpertOutputs`: `batch seq top_k hidden` | `weighted_out.view(tokens, top_k, hidden)` in transformers' `grouped_mm` / `batched_mm` experts forward |
| `routed_output` | the routed experts' sum, without the shared expert | `Residual` | the experts module's output |
| `shared_expert_output` | the shared expert's contribution, where the mixture has one | `Residual` | the shared expert's output (`shared_experts`; aliased from `shared_expert` / `shared_mlp`) |

The router is aliased `router` (from `gate` where the family calls it that) and the shared
expert `shared_experts`, so `moe.router.output` and `moe.shared_experts.output` reach the
native modules too. `moe.num_experts` and `moe.top_k` are read off the router or the
experts module, like `mlp.intermediate_size`. `Moe.SCORING` says what the logits mean:

| `SCORING` | the router | families |
| --- | --- | --- |
| `"softmax"` | softmax over every expert, then top-k (renormalized or not, per family) | Mixtral, Qwen2/3-MoE, Qwen3-Next, Qwen3.5-MoE, OLMoE, FlexOlmo, DeepSeek-V2, Hunyuan, ERNIE, Jamba, DBRX, Gemma-4, ZAYA |
| `"topk_softmax"` | top-k of the logits, then a softmax over those | GPT-OSS, GraniteMoE (-Shared, -Hybrid, -SWA), JetMoE |
| `"sigmoid"` | a sigmoid per expert (plus a selection bias on most) | DeepSeek-V3/V3.2, Kimi K2, Kimi-Linear, GLM-4-MoE(-Lite), GLM-5, dots.llm1, Solar Open, MiMo-V2-Flash, Nemotron-H, MiniMax-M2, Laguna, AFMoE, Llama 4 |
| `"sparsemixer"` | Phi-3.5-MoE's masked softmax per slot | Phi-3.5-MoE |
| `"hash"` / the config's `scoring_func` | DeepSeek-V4, per block: the token id picks the experts on a `hash_moe` block; elsewhere `scoring_func` (`"sqrtsoftplus"`) per expert | DeepSeek-V4 |

`SCORING` names the function, not everything between the logits and `expert_weights`.
Recomputing the weights from `router_logits` also needs the family's normalization of the
top-k, a config flag: OLMoE (and Qwen1.5-MoE) leave `norm_topk_prob` off, so the weights
are the softmax's top-k entries themselves and a token's weights sum to less than one
(0.33 to 0.77 on one block of OLMoE-1B-7B), while Mixtral renormalizes them to one. Some
families also scale the weights after (`routed_scaling_factor`): DeepSeek-V2 leaves
`norm_topk_prob` off and multiplies the chosen softmax entries by `routed_scaling_factor`,
16 on the 236B checkpoints (V2, V2-Chat, V2.5) and 1.0 on V2-Lite, so its weights are the
softmax entries times that factor. Read `expert_weights`
rather than recomputing it where you can.

## Recipes

Each runs as written on the tiny Mixtral above, and on every MoE family with the values it
has ([below](#per-family-caveats)), JetMoE needing `expert_indices` read before
`expert_weights` ([read order](#read-order-within-a-mixture)).

```python
prompts = ["The Eiffel Tower is in the city of", "def add(a, b):\n    return a + b"]
with model.trace(prompts):                   # a batch is left-padded, and the router routes the pads too
    mask = model.attention_mask.save()       # [batch, seq]: 0 on pad positions
    logits = moe.router_logits.save()
    w = moe.expert_weights.save()
    idx = moe.expert_indices.save()
real = mask.bool()                           # count real tokens only

# Expert usage: how many slots each expert got (mask w == 0: ZAYA's skipped slots read as expert 0)
usage = torch.bincount(idx[real][w[real] != 0], minlength=moe.num_experts)

# Routing entropy per real token, on a softmax router (`SCORING == "softmax"`)
probs = logits[real].float().softmax(-1)
entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1)          # [real tokens]

prompt = "The Eiffel Tower is in the city of"

# Ablate expert 3 everywhere: zero the weight of every slot that chose it
with model.trace(prompt):
    moe.expert_weights = moe.expert_weights.masked_fill(moe.expert_indices == 3, 0)
    ablated = model.logits.save()
# (JetMoE computes the indices first: idx = moe.expert_indices, then masked_fill(idx == 3, 0))

# Ablate one slot: the last token's first choice, in place
with model.trace(prompt):
    moe.expert_weights[:, -1, 0] = 0

# Reroute the last token's first slot to expert 2 (its weight stays what the router gave the expert it chose)
with model.trace(prompt):
    rerouted = moe.expert_indices.clone()
    rerouted[:, -1, 0] = 2
    moe.expert_indices = rerouted

# Force the router: logits written before the scoring decide the weights and the indices
with model.trace(prompt):
    moe.router_logits[:, -1] = torch.tensor([-10.0, -10.0, 10.0, 5.0])
    forced = moe.expert_indices.save()      # the last token routes to [2, 3]
```

Zeroing a slot's weight removes exactly that slot's term from `routed_output`; the suite
checks it against `expert_outputs` on every family. A rerouted slot computes
`w * expert_e2(x)`; the other tokens are untouched up to the grouped matmul's rounding
(their groups change size). An edit also reaches the routing of later blocks: a token
whose stream changes can pick other experts downstream, so compare edited runs with a
clean run in float32, where the rest of the forward rounds the same way.
[expert-ablation](../patterns/expert-ablation.md) sweeps every
expert's effect on a target token.

## `experts_implementation=` and what eager loses

transformers runs the routed experts through one of several implementations, chosen at
load like `attn_implementation`: `"grouped_mm"` (the default), `"batched_mm"` or
`"eager"` (a Python loop over the experts), or a hub kernel. The per-slot outputs exist as
one tensor only in `grouped_mm` and `batched_mm`, so `expert_outputs` is unavailable under
the others, with a reason naming the kwarg:

```python
eager = StandardizedTransformer("hf-internal-testing/tiny-random-MixtralForCausalLM", dispatch=True, experts_implementation="eager")
eager.support(layer=1)["mlp.expert_outputs"]
# "read inside transformers' grouped_mm / batched_mm experts forward, but this model runs 'eager';
#  load with experts_implementation='grouped_mm' (the default) or 'batched_mm'"
```

`support()` reads the implementation off the model's config, so on a lazy load it is
right once the weights are in (`dispatch=True`, or after the first trace), or before
that where nnsight's meta build forwards `experts_implementation=`.
Everything else reads and writes the same under every implementation: the routing pair
and the routed sum are module boundaries. transformers also picks `eager` by itself where
`grouped_mm` cannot run (CUDA below SM80). Unweighted per-slot outputs are
`expert_outputs / expert_weights[..., None]` where the weight is not zero.

## Read order within a mixture

Reads in one trace follow the forward: `router_logits`, then `expert_weights` /
`expert_indices`, then `expert_outputs`, then `routed_output`, then `mlp_output`. Where the
shared expert runs differs:

| `shared_expert_output` comes | families |
| --- | --- |
| first, before the router | Hunyuan, ERNIE, Laguna, Gemma-4 (the dense MLP runs first) |
| after `router_logits`, before `expert_weights` / `expert_indices` (read at the experts' arguments) | AFMoE |
| after `routed_output` | DeepSeek-V2/V3/V3.2/V4, GLM-4-MoE(-Lite), GLM-5, dots.llm1, Solar Open, Nemotron-H, the Qwen families (the gated product), GraniteMoE-Shared and -Hybrid |
| after the routing, before `routed_output` | Llama 4 |

On Qwen2-MoE, Qwen3-Next and Qwen3.5-MoE only the gated product (`shared_expert_output`)
comes after `routed_output`: the ungated shared expert (`mlp.shared_expert.output`) runs
first, before `router_logits`. On JetMoE the router computes the indices before the
weights, so read `expert_indices` before `expert_weights` (the ablation below binds the
indices first there).

An out-of-order read raises nnsight's `OutOfOrderError`; read one value per trace when
in doubt, as the suite does.

## Batches and invokes

The model routes tensors flat over tokens, `[batch * seq, ...]`. Each value is a `TokenEProperty`, served as
`[batch, seq, ...]`: a whole batch in one invoke, this invoke's rows under several. The
rows are a view, so in-place edits land on the model's tensor and reach that invoke only.
An assignment under one invoke always lands; under two or more it needs nnsight's widen
of an edit to a tensor whose leading axis is not the batch (nnsight PR #738). On an nnsight
without it, edit in place under several invokes.

## Per-family caveats

| family | what differs |
| --- | --- |
| Gemma-4 (26B-A4B) | No MoE module: `router` and `experts` are the block's children; `layers[i].mlp`, the dense MLP, reads them through its parent and hosts the values. The router runs on the stream after the attention's add (not the block's input) with its own norm and returns probabilities; `router_logits` is its projection (`router.proj`). `shared_expert_output` is the dense MLP's output; the identity is `mlp_output == post_feedforward_layernorm(post_feedforward_layernorm_1(shared_expert_output) + post_feedforward_layernorm_2(routed_output))`. On a dense checkpoint every mixture value is unavailable. |
| GraniteMoE-Hybrid | `mlp` is the shared expert (`shared_mlp`); it reads `block_sparse_moe`'s `router` and `experts` through its parent. `shared_expert_output` is `mlp.output`; `routed_output + shared_expert_output == mlp_output / residual_multiplier`. GraniteMoE-Shared: `shared_expert_output` is the block's `shared_mlp`, the same identity. |
| Llama 4 | The router scatters the sigmoid of the top-k logits into dense scores over every expert, every expert runs on every token scaled on its *input* by its score, and the routed sum is added into the shared expert's output tensor in place. `router_logits` and `expert_indices` (the router's top-k) are served, `routed_output` is the sum over experts, `shared_expert_output` a copy carried back by a transform; `expert_weights` and `expert_outputs` are unavailable. |
| JetMoE | No experts module: the router takes the top-k of its logits, softmaxes them and sorts the slots by expert. `expert_indices` / `expert_weights` are its top-k indices and gates in token order; `routed_output` is the routed sum before the mixture's `+ bias`; `expert_outputs` (sorted by expert) is unavailable. Read order: logits, indices, weights. |
| FlexOlmo | The released checkpoints route every token to every expert (`num_experts_per_tok == num_experts`) with `norm_topk_prob` off: `expert_weights` is the whole softmax, summing to one, and ablating an expert's slot removes its share without rerouting. |
| DBRX | The router (`router.layer`) returns the logits alone; the FFN's `route_tokens_to_experts` takes a softmax top-k and p-normalizes. The experts loop over experts in their own forward, so `expert_outputs` is unavailable. |
| ZAYA | `router_logits` has `num_experts + 1` columns: the last is **skip**. A slot that picks it has weight 0 and index **0**, an alias of expert 0, so mask `expert_weights != 0` when counting usage. On ZAYA1-8B and ZAYA1-74B-preview (top 1) the skip class's balancing bias is -1 and some expert's is positive on every block, so skip is never chosen there; the tiny picks it. The router carries a state from the previous block. |
| Nemotron-H | With `moe_latent_size` the experts run in a latent width between `fc1_latent_proj` and `fc2_latent_proj`: `routed_output` is the up projection's output and `expert_outputs` is unavailable. |
| DeepSeek-V4 | On a `hash_moe` block the token ids pick the experts (`tid2eid[input_ids]`): writing `router_logits` changes `expert_weights`, not `expert_indices`. `SCORING` is per block. |
| Laguna | `router_logits` is before the router's tanh softcap; `routed_output` is the experts' sum times `routed_scaling_factor`, so `expert_outputs.sum(2) * routed_scaling_factor == routed_output`. |
| Qwen2-MoE, Qwen3-Next, Qwen3.5-MoE | `shared_expert_output` is the shared expert's output times its sigmoid gate, the product the mixture adds; `moe.shared_experts.output` is the ungated one. |
| GraniteMoE, -SWA, -Shared, -Hybrid | `mlp_output` is the scaled term the block adds, the mixture's output times `residual_multiplier` (0.22 on granite-3.0-1b-a400m); `expert_outputs`, `routed_output` and `shared_expert_output` are unscaled, about 4.5 times what reaches the stream there. Multiply by `model.config.residual_multiplier` to compare them with `mlp_output` or another family's experts. |
| Doge | Its cross-domain mixture (`is_moe`) cannot run in transformers 5.17 (the block drops out its tuple output), so every mixture value is unavailable. |

`model.support()` lists the six values under `mlp.`; on a family with dense blocks beside
mixture blocks (DeepSeek-V3, GLM-4-MoE, Llama 4, Jamba, ...), a dense block reports
`"no <value> value on this block's mlp"`.

## Related

- [residual-stream](residual-stream.md) — `mlp_output` and the contribution identity.
- [layouts](layouts.md) — `RouterLogits`, `ExpertWeights`, `ExpertIndices`, `ExpertOutputs`.
- [availability](availability.md) — `support()` and the reasons.
- [expert-ablation](../patterns/expert-ablation.md) — every expert's effect on a prediction.
- [families](../reference/families.md) — the mixture-of-experts families and their overrides.
