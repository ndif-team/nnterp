# Expert ablation

Removing one routed expert asks whether a prediction depends on it: zero `expert_weights` wherever `expert_indices == e`, and the slots that chose expert `e` add nothing while the other slots keep their weights, so the token is not renormalized onto its remaining experts and the router is not re-run. One invoke per expert gives a `[block, expert]` grid of changes in log p(target) with one trace per block, the same code on every MoE family ([Mixture of experts](../guide/architectures/mixture-of-experts.md)). Reading the routing rather than editing it is nnsight.net's [MoE Routing](https://nnsight.net/tutorials/tutorials/architectures/moe_routing/) tutorial.

## The grid

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("ibm-granite/granite-3.0-1b-a400m-base", dispatch=True, device="cuda", dtype=torch.float32)
prompt = "Big Ben is in the city of"
ids = model.tokenizer(" London", add_special_tokens=False).input_ids
assert len(ids) == 1, model.tokenizer.convert_ids_to_tokens(ids)       # one token, or the target is not the word
target = ids[0]
blocks = [layer.mlp for i, layer in enumerate(model.layers) if model.support(layer=i).get("mlp.expert_weights", "absent") is None]
print(len(blocks), blocks[0].num_experts, blocks[0].top_k)             # 24 32 8


def sweep(exclude_first_token=False):
    """[block, expert]: the change in log p(target) when expert e's slots are zeroed."""
    effects = torch.zeros(len(blocks), blocks[0].num_experts)
    for b, moe in enumerate(blocks):
        rows = []                                                      # bound outside the trace
        with torch.no_grad(), model.trace() as tracer:
            with tracer.invoke(prompt):                                # unedited: the baseline, from the same batch
                clean = model.logits[0, -1].log_softmax(-1)[target].save()
            for e in range(moe.num_experts):
                with tracer.invoke(prompt):
                    chosen = moe.expert_indices == e                   # [batch, seq, top_k]
                    if exclude_first_token:
                        chosen[:, 0] = False
                    moe.expert_weights[:] = moe.expert_weights.masked_fill(chosen, 0)
                    rows.append(model.logits[0, -1].log_softmax(-1)[target].save())
        effects[b] = torch.stack(rows) - clean
    return effects, float(clean)


def top(effects, k=4):
    for flat in effects.flatten().topk(k, largest=False).indices:
        b, e = divmod(int(flat), effects.shape[1])
        print(f"block {b:2d} expert {e:2d}  {float(effects[b, e]):+.3f}")


effects, clean = sweep()
print(f"clean log p(London) {clean:.3f}; experts no token chose: {int((effects == 0).sum())} of {effects.numel()}")
top(effects)
# clean log p(London) -0.359; experts no token chose: 205 of 768
# block  5 expert  4  -9.711
# block  0 expert  4  -2.247
# block  5 expert 17  -0.820
# block  4 expert  2  -0.555
```

`support(layer=i)` picks the blocks with the routing pair. The dense blocks beside the mixture blocks on DeepSeek-V3, GLM-4-MoE, Llama 4 and Jamba have none. `add_special_tokens=False` keeps the BOS token out of the target, and the assertion catches a word the tokenizer splits: this one splits ` Paris` into `Par` and `is`, and the target is a city it spells in one token. Under several invokes the edit is in place (`expert_weights[:] = ...`), because each invoke is served its own rows of the flat routing tensors, and the baseline is the unedited invoke of the same batch rather than a separate single-prompt trace, since a batch rounds differently from one prompt. Load in float32 for a single-expert sweep: the median effect here is 0.009 nats, and in bfloat16 the clean run alone differs between a batch and a single prompt by 0.018 nats (by 1e-6 in float32).

An expert no token of the prompt chose has no slots to zero, and its cell is exactly zero: 205 of the 768 cells here. A zero marks an unused expert, not an unimportant one.

## The first token

```python
with torch.no_grad(), model.trace(prompt):
    chosen = (blocks[5].expert_indices[0] == 4).nonzero().save()      # [slot, (position, k)]
    weights = blocks[5].expert_weights[0].save()
print(chosen.tolist(), [round(float(weights[p, k]), 3) for p, k in chosen])   # [[0, 0], [3, 3]] [0.997, 0.073]

top(sweep(exclude_first_token=True)[0])
# block  0 expert  4  -2.247
# block  5 expert 17  -0.820
# block  4 expert  2  -0.572
# block  0 expert 22  -0.516
```

Block 5's expert 4 costs 9.7 nats, and it takes the first token with weight 0.997 and the fourth with 0.073. The effect comes through position 0 alone, and with that position excluded it drops off the list. Rank experts on positions 1 onward, or ablate per position, before reading a large effect as a mechanism. The ranking that survives puts block 0's expert 4 first at 2.2 nats, then block 5's expert 17 and block 4's expert 2 under one nat each, which is the scale of a single expert's contribution to this prediction.

An edit to one token's stream also changes that token's routing at later mixture blocks, so an effect includes the rerouting it causes. As with any zero ablation, the number overstates what the expert does when present. Rerouting its slots to another expert, by assigning `expert_indices`, is the milder intervention.

## On another family

`expert_weights` and `expert_indices` are `[batch, seq, top_k]` on every MoE family and `num_experts` and `top_k` are read off the mixture, so the grid's shape follows the checkpoint and the code does not change. A mixture's shared expert is not among the routed ones; ablate it with `shared_expert_output[:] = 0` where `support()` lists it. On JetMoE the indices are computed before the weights, so bind `idx = moe.expert_indices` before reading `expert_weights`, and mask with `idx == e`.
