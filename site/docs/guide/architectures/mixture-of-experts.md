# Mixture of experts

On a mixture-of-experts checkpoint `layers[i].mlp` is a `Moe`, an `Mlp` that also serves the routing and the experts: `router_logits`, `expert_weights`, `expert_indices`, `expert_outputs`, `routed_output` and `shared_expert_output`. `mlp_output` stays what the block adds to the residual stream, as on every MLP. The smallest public checkpoint with a mixture that nnterp can read is ibm-granite/granite-3.0-1b-a400m-base, 32 experts and 8 slots per token. The names are the same on all 39 MoE families.

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("ibm-granite/granite-3.0-1b-a400m-base", device="cuda", dtype=torch.float32)
moe = model.layers[3].mlp
print(moe.num_experts, moe.top_k)

with model.trace("The Eiffel Tower is in the city of"):
    logits = moe.router_logits.save()
    weights = moe.expert_weights.save()
    indices = moe.expert_indices.save()
    slots = moe.expert_outputs.save()
    routed = moe.routed_output.save()

print(tuple(logits.shape), tuple(weights.shape), tuple(indices.shape), tuple(slots.shape), tuple(routed.shape))
print(indices[0, -1].tolist())
print(torch.allclose(slots.sum(2), routed, atol=1e-5))
# 32 8
# (1, 11, 32) (1, 11, 8) (1, 11, 8) (1, 11, 8, 1024) (1, 11, 1024)
# [2, 0, 10, 30, 20, 16, 18, 22]
# True
```

## The values

| value | shape | what it is |
| --- | --- | --- |
| `router_logits` | `[batch, seq, num_experts]` | the router's logits before the scoring (a softmax, a top-k softmax or a sigmoid, named by `moe.SCORING`); writable, and the router scores what is written |
| `expert_weights` | `[batch, seq, top_k]` | what multiplies each slot's expert output, after the router's normalization and scaling |
| `expert_indices` | `[batch, seq, top_k]`, int64 | the expert each slot sends the token to |
| `expert_outputs` | `[batch, seq, top_k, hidden_size]` | each slot's expert output times its weight; `expert_outputs.sum(2) == routed_output` |
| `routed_output` | `[batch, seq, hidden_size]` | the routed experts' sum, without the shared expert |
| `shared_expert_output` | `[batch, seq, hidden_size]` | the shared expert's contribution, where the mixture has one |

A slot is one of a token's `top_k` routing choices. The weights and indices are read as the experts module receives them, so a write there is what the experts compute with. `moe.num_experts` and `moe.top_k` are read off the module. On Granite the block multiplies each sublayer's output by `residual_multiplier`, 0.22 here, and `mlp_output` is `routed_output * 0.22`. The experts' values are unscaled on every family. Signatures are on the [`moe` page](../../documentation/components/moe.md) of the Documentation tab.

## Ablating and rerouting

```python
prompt = "The Eiffel Tower is in the city of"
with model.trace(prompt):
    clean = model.logits[0, -1].save()

with model.trace(prompt):                       # expert 2 out of every slot that chose it
    moe.expert_weights = moe.expert_weights.masked_fill(moe.expert_indices == 2, 0)
    ablated = model.logits[0, -1].save()

with model.trace(prompt):                       # the last token's first slot goes to expert 3 instead
    rerouted = moe.expert_indices.clone()
    rerouted[:, -1, 0] = 3
    moe.expert_indices = rerouted
    rerouted_weights = moe.expert_weights.save()
    moved = model.logits[0, -1].save()

with model.trace(prompt):                       # written before the scoring, the logits decide the routing
    forced = torch.full((moe.num_experts,), -10.0, device="cuda")
    forced[[2, 5]] = 10.0
    moe.router_logits[:, -1] = forced
    chosen = moe.expert_indices.save()

print((ablated - clean).abs().max().item(), (moved - clean).abs().max().item())
print(torch.equal(rerouted_weights, weights), chosen[0, -1, :2].tolist())
# 0.7101480960845947 1.5887203216552734
# True [5, 2]
```

Zeroing a weight removes exactly that slot's term from `routed_output`. A rerouted slot computes `weight * expert_3(x)` with the weight the router gave the expert it replaced, and nothing is renormalized. A write to `router_logits` lands before the scoring, and the weights and indices follow it. The two forced experts share the weight, 0.5 each under Granite's top-k softmax.

!!! warning "Take the baseline from the same run"
    An edit to one block's routing changes the stream. Later blocks can then send the token to other experts, and the rest of the forward rounds differently, which is why the clean run above is in float32, where the unedited part rounds the same way. Under two or more invokes the batch is padded and rounds differently from the single prompt (`base` below is not `clean`). Edit the routing in place there and take the baseline from an unedited invoke of the same batch:

    ```python
    with model.trace() as tracer:
        with tracer.invoke(prompt):
            base = model.logits[0, -1].save()
        with tracer.invoke(prompt):
            moe.expert_weights.masked_fill_(moe.expert_indices == 2, 0)
            edited = model.logits[0, -1].save()

    print(torch.equal(base, clean), (edited - base).abs().max().item())
    # False 0.7101457118988037
    ```

## Read order

Within a trace, read a mixture's values in forward order: `router_logits`, then `expert_weights` and `expert_indices`, then `expert_outputs`, `routed_output` and `mlp_output`. A read out of order raises nnsight's `OutOfOrderError`. Where `shared_expert_output` falls differs by family: after `routed_output` on DeepSeek-V3 and on Qwen2-MoE, Qwen3-Next and Qwen3.5-MoE (where it is the shared expert's output times its sigmoid gate), before the router on Hunyuan, ERNIE and Gemma-4. JetMoE's router computes the indices before the weights, and `expert_indices` comes first there. Granite has no shared expert, and `support()` says so:

```python
print(model.support(layer=3)["mlp.shared_expert_output"])
# this mixture has no shared expert
```

## Pad tokens

The router routes pad tokens like any other. When counting usage over a batch, keep the real positions with `model.attention_mask`:

```python
prompts = ["The Eiffel Tower is in the city of", "def add(a, b):\n    return a + b"]
with model.trace(prompts):
    mask = model.attention_mask.save()          # [batch, seq], 0 at the pad positions
    indices = moe.expert_indices.save()

real = mask.bool()
usage = torch.bincount(indices[real].flatten(), minlength=moe.num_experts)
print(mask[0].tolist())
print(usage.tolist())
# [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# [9, 8, 14, 0, 9, 1, 5, 6, 1, 4, 2, 1, 4, 13, 0, 0, 12, 0, 7, 1, 3, 3, 11, 5, 12, 2, 15, 10, 3, 3, 12, 8]
```

## `expert_outputs` needs the grouped experts

transformers runs the routed experts through one of several implementations, chosen at load like `attn_implementation`: `grouped_mm` (the default), `batched_mm`, `eager` (a Python loop over the experts) or a hub kernel. The per-slot outputs exist as one tensor only under the first two. Under the others `expert_outputs` is unavailable, and `support()` names the kwarg. The routing pair, `routed_output` and `mlp_output` are module boundaries and read the same under every implementation.

```python
eager = StandardizedTransformer("ibm-granite/granite-3.0-1b-a400m-base", device="cuda", experts_implementation="eager")
print(eager.support(layer=3)["mlp.expert_outputs"])
# read inside transformers' grouped_mm / batched_mm experts forward, but this model runs 'eager'; load with experts_implementation='grouped_mm' (the default) or 'batched_mm'
```

[Expert ablation](../../tutorials/expert-ablation.md) sweeps every expert's effect on a prediction.
