# Attention

Six values on `model.layers[i].self_attn` are what happens inside the attention sublayer: the queries, keys and values that enter the softmax attention, the scores entering the softmax, the pattern the values are mixed with, and each head's output before the heads are concatenated and projected. They are read inside transformers' eager attention forward, so the model has to be loaded with `attn_implementation="eager"`.

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", attn_implementation="eager")
attn = model.layers[5].self_attn

with model.trace("The Eiffel Tower is in"):
    q = attn.attention_queries.save()
    k = attn.attention_keys.save()
    v = attn.attention_values.save()
    scores = attn.attention_scores.save()
    pattern = attn.attention_probabilities.save()
    heads = attn.attention_head_outputs.save()

print(q.shape, k.shape, v.shape)
# torch.Size([1, 12, 7, 64]) torch.Size([1, 12, 7, 64]) torch.Size([1, 12, 7, 64])
print(scores.shape, pattern.shape, heads.shape)
# torch.Size([1, 12, 7, 7]) torch.Size([1, 12, 7, 7]) torch.Size([1, 7, 12, 64])
print(torch.allclose(scores.softmax(-1), pattern), pattern[0, 0].sum(-1))
# True tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
#        grad_fn=<SumBackward1>)
```

| value | shape | what it is |
| --- | --- | --- |
| `attention_queries` | `[batch, num_heads, seq, qk_head_dim]` | the queries entering attention, after the query projection and, where the family has one, the rotary embedding |
| `attention_keys` | `[batch, num_kv_heads, seq, qk_head_dim]` | the keys entering attention, before `repeat_kv` |
| `attention_values` | `[batch, num_kv_heads, seq, head_dim]` | the values entering attention, before `repeat_kv` |
| `attention_scores` | `[batch, num_heads, query, key]` | the scaled scores entering the softmax, with the causal mask applied |
| `attention_probabilities` | `[batch, num_heads, query, key]` | the pattern the values are mixed with: after the softmax and its dropout, in the model's dtype |
| `attention_head_outputs` | `[batch, seq, num_heads, head_dim]` | each head's mix of the values, before the output projection |

The axis names are the sizes on the attention envoy, `attn.num_heads`, `attn.num_kv_heads`, `attn.head_dim` and `attn.qk_head_dim`, read off the module ([Layouts](layouts.md)). `qk_head_dim` equals `head_dim` outside latent attention (DeepSeek-V2 and V3). The pattern's rows sum to one, except on a model with an attention sink (GPT-OSS, MiMo-V2-Flash, DeepSeek-V4), where the sink's column has already been dropped and they sum to less.

## Eager attention

A load without `attn_implementation` gets transformers' default, `sdpa` on most families, and the eager forward never runs. `model.support()` gives the reason per value, and reading one raises `nnterp.Unavailable` with the same reason before the model runs. `attention_output` and `attention_premix` (`o_proj.input`, the heads as the output projection receives them) stay available under any implementation.

```python
default = StandardizedTransformer("openai-community/gpt2")
print(default.support(layer=5)["self_attn.attention_probabilities"])
# read inside the eager attention forward, but this model runs 'sdpa'; load with attn_implementation='eager'
print(default.support(layer=5)["self_attn.attention_output"])
# None
```

## Editing

Each value is the tensor the forward holds, so an in-place edit lands and an assignment replaces that one value: assigning the keys leaves the queries and values as they were. GPT-2's and GPT-BigCode's queries are the exception: `c_attn` produces one tensor that `split` divides into queries, keys and values, and torch refuses an in-place edit of the query view (`RuntimeError: Output 0 of Select is a view and is being modified inplace`), so assign them. Their keys and values take in-place edits.

```python
with model.trace("The Eiffel Tower is in"):
    clean = model.logits.save()

with model.trace("The Eiffel Tower is in"):
    attn.attention_head_outputs[:, :, 7] = 0
    ablated = model.logits.save()

with model.trace("The Eiffel Tower is in"):
    attn.attention_queries = attn.attention_queries * 0
    attn.attention_keys[:, :, 0] = 0
    replaced = model.logits.save()

print((ablated - clean).abs().max().item(), (replaced - clean).abs().max().item())
# 2.7936935424804688 13.05731201171875
```

!!! warning "Overwriting or multiplying `attention_scores` lifts the causal mask"
    The masked entries hold the dtype's minimum (`-inf` on GPT-Neo). `scores[:] = 0` or `scores * 0` turns them into real scores, every query then attends to the tokens after it, and the pattern still looks like a valid distribution. Add to the scores instead, or keep the masked entries: `torch.where(scores <= torch.finfo(scores.dtype).min, scores, new)`.

```python
with model.trace("The Eiffel Tower is in"):
    attn.attention_scores[:, :, :, 0] += 5.0
    added = attn.attention_probabilities.save()

with model.trace("The Eiffel Tower is in"):
    attn.attention_scores[:] = 0
    overwritten = attn.attention_probabilities.save()

print(added[0, 0, 2])
# tensor([9.9987e-01, 8.8083e-05, 4.0907e-05, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#         0.0000e+00], grad_fn=<SelectBackward0>)
print(overwritten[0, 0, 2])
# tensor([0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429],
#        grad_fn=<SelectBackward0>)
```

## Grouped-query attention

The keys and values are read before `repeat_kv`, so on a checkpoint with fewer key/value heads than query heads their head axis is `num_kv_heads` wide while the queries, the scores, the pattern and the head outputs are `num_heads` wide. Query head `h` reads key/value head `h // (num_heads // num_kv_heads)`, so a head-wise edit of the keys touches every query head of that group. On Qwen2.5-0.5B, 14 query heads share 2 key/value heads:

```python
qwen = StandardizedTransformer("Qwen/Qwen2.5-0.5B", attn_implementation="eager")
attn = qwen.layers[5].self_attn
print(attn.num_heads, attn.num_kv_heads, attn.head_dim)
# 14 2 64

with qwen.trace("The Eiffel Tower is in"):
    k = attn.attention_keys.save()
    v = attn.attention_values.save()
    pattern = attn.attention_probabilities.save()
    heads = attn.attention_head_outputs.save()

print(k.shape, pattern.shape, heads.shape)
# torch.Size([1, 2, 7, 64]) torch.Size([1, 14, 7, 7]) torch.Size([1, 7, 14, 64])
groups = attn.num_heads // attn.num_kv_heads
mixed = (pattern @ v.repeat_interleave(groups, dim=1)).transpose(1, 2)
print(torch.allclose(mixed, heads, atol=1e-5))
# True
```

## Read order

The table lists the six in forward order, and all of them come before `attention_output`. Several blocks' values in one trace are fine in ascending block order; the rule is forward order across the whole model, not only within a block. `model.order(i)` gives the order for block `i`, with equal ranks for values read at one location:

```python
print(qwen.order(5))
# {'layer_input': 0, 'attention_norm_output': 1, 'self_attn.attention_queries': 2, 'self_attn.attention_keys': 2, 'self_attn.attention_values': 2, 'self_attn.attention_scores': 3, 'self_attn.attention_probabilities': 4, 'self_attn.attention_head_outputs': 5, 'self_attn.attention_premix': 6, 'self_attn.attention_output': 7, 'layer_mid': 8, 'mlp_norm_output': 9, 'mlp_input': 10, 'mlp.mlp_activation': 11, 'mlp.mlp_neurons': 12, 'mlp.mlp_output': 13, 'layer_output': 14}
```

A read out of order raises nnsight's `OutOfOrderError`, which names the internal location rather than the value:

```python
attn = model.layers[5].self_attn
try:
    with model.trace("The Eiffel Tower is in"):
        out = attn.attention_output.save()
        pattern = attn.attention_probabilities.save()
except Exception as e:
    print(type(e).__name__, e)
# OutOfOrderError 'model.transformer.h.5.attn.source.attention_interface_1.fn.i0' was requested but the model already ran past it
```

Falcon's attention does its own arithmetic, and without alibi its values bind before the queries and keys leave the rotary embedding, so read `attention_values` first there; with alibi the order is queries, keys, values.

Under `model.generate` the values are per forward call: on a decode step the queries, the scores, the pattern and the head outputs have one query position, and the keys, the values and the pattern's key axis are as long as the cache ([Generation](../generation.md)):

```python
with model.generate("The Eiffel Tower is in", max_new_tokens=2) as tracer:
    for _ in tracer.iter[1:2]:
        pattern = attn.attention_probabilities.save()
print(pattern.shape)
# torch.Size([1, 12, 1, 8])
```

The signatures are on [`Attention`](../../documentation/components/attention.md), and the values of the whole model, from the input to the logits, are on [The whole model](whole-model.md).
