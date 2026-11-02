# Ablation

Removing a contribution, a head or a block and measuring the effect.

Zero `attention_output` or `mlp_output` and the sublayer adds nothing to the residual stream, on every family, including the families that add the residual inside the module (BLOOM, MPT) or norm the output first (Gemma-2, Gemma-3, OLMo-2). One head is one slice of `attention_head_outputs`, and whole blocks are `model.skip_layers`. The clean and ablated runs are invokes of one trace, so one forward gives the comparison. nnsight's [batching](https://nnsight.net/features/8_batching/) page covers invokes.

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", device="cpu", attn_implementation="eager")
prompt = "The Eiffel Tower is in the city of"
paris = model.tokenizer(" Paris", add_special_tokens=False).input_ids[0]
LAYER, HEAD = 9, 8
block = model.layers[LAYER]

runs = {}                                                   # bound outside the trace
with torch.no_grad(), model.trace() as tracer:
    with tracer.invoke(prompt):
        runs["clean"] = model.next_token_probs[0, paris].save()
    with tracer.invoke(prompt):
        block.self_attn.attention_output[:] = 0             # this invoke's rows only
        runs["attention"] = model.next_token_probs[0, paris].save()
    with tracer.invoke(prompt):
        block.mlp.mlp_output[:] = 0
        runs["mlp"] = model.next_token_probs[0, paris].save()
    with tracer.invoke(prompt):
        block.self_attn.attention_head_outputs[..., HEAD, :] = 0
        runs[f"head {HEAD}"] = model.next_token_probs[0, paris].save()

with torch.no_grad(), model.trace(prompt):                  # a skip covers every row, so it gets its own trace
    model.skip_layers(LAYER, LAYER + 1)
    runs["blocks 9-10"] = model.next_token_probs[0, paris].save()

for name, p in runs.items():
    print(f"{name:12s} P(Paris) {float(p):.3f}")
# clean        P(Paris) 0.070
# attention    P(Paris) 0.033
# mlp          P(Paris) 0.053
# head 8       P(Paris) 0.036
# blocks 9-10  P(Paris) 0.006
```

Block 9's attention carries half of P(Paris) (0.070 to 0.033), and one head, head 8, accounts for almost all of that (0.036). The block's MLP matters less (0.053). Skipping blocks 9 and 10 leaves 0.006. The writes are in place on the served tensors, so they reach the model, and a write in one invoke touches only that invoke's rows. `skip_layers` is in a trace of its own because a skip has to cover every row of a forward. Inside one invoke of a batched trace it raises `ValueError: A batched .skip() has to cover every row`.

## Mean ablation

Replacing a contribution with its mean over a reference set, instead of zero, keeps the stream nearer to its usual state and asks whether the deviation from average matters:

```python
reference = ["The capital of France is", "Berlin lies in", "London is the capital of", "Tokyo is in"]
with torch.no_grad(), model.trace(reference):
    mean = model.layers[LAYER].mlp.mlp_output[:, -1].mean(0).save()     # [hidden_size]

with torch.no_grad(), model.trace(prompt):
    model.layers[LAYER].mlp.mlp_output[:, -1] = mean
    p = model.next_token_probs[0, paris].save()
print(f"mean-ablated P(Paris) {float(p):.3f}")
# mean-ablated P(Paris) 0.073
```

Zeroing block 9's MLP costs 0.017 of P(Paris), and replacing it with its mean costs nothing. The two answer different questions, so say which you ran. A list in one trace is left-padded, which is what makes `[:, -1]` every reference prompt's last token.

## Every head of a block

One invoke per head, in one forward:

```python
rows = []
with torch.no_grad(), model.trace() as tracer:
    for head in range(model.num_heads):
        with tracer.invoke(prompt):
            model.layers[LAYER].self_attn.attention_head_outputs[..., head, :] = 0
            rows.append(model.next_token_probs[0, paris].save())
print([round(float(p), 3) for p in rows])
# [0.077, 0.07, 0.068, 0.065, 0.065, 0.066, 0.069, 0.072, 0.036, 0.073, 0.069, 0.07]
```

Head 8 is the only head of block 9 with an effect. The others move P(Paris) by under 0.01 in either direction. Zeroing a head's rows of `attention_probabilities` gives the same logits as zeroing its slice here. Both need `attn_implementation="eager"`. The boundary values `attention_output`, `mlp_output` and `layer_output` do not. Do not slice `attention_output` into `head_dim`-wide column blocks instead, because after the output projection the hidden axis no longer decomposes per head.

## On another family

The contribution values do the per-family work, so the recipe runs unchanged wherever the block has the module. OPT has no `mlp` module, and a hybrid's DeltaNet blocks have `linear_attn` instead of `self_attn`, so decide which blocks to ablate from `model.support()` or the module list, outside the trace. On Granite the block adds `residual_multiplier` times the sublayer's output, and nnterp serves the scaled copy, so a write at one position moves the other positions by rounding in bf16. Load in float32 for small edits there. The meaning of the values is in [Residual stream](../guide/values/residual-stream.md).
