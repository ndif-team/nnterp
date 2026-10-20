# Layouts

Every value has one axis layout, the same on every family: `attention_probabilities` is `[batch, num_heads, query, key]` on GPT-2, Llama and BLOOM alike. The layout is a named `jaxtyping` type from `nnterp.components` (`Residual`, `Pattern`, `Queries`, ...), read off the value as `value.layout`, with its axes as `value.dims`.

```python
from nnterp import Attention, Layer, StandardizedTransformer
from nnterp.components import Pattern, Queries, Residual

print(Layer.layer_output.dims, Layer.layer_output.layout is Residual)
print(Attention.attention_queries.dims)

model = StandardizedTransformer("openai-community/gpt2", dispatch=True, device="cpu", attn_implementation="eager")
with model.trace("The Eiffel Tower is in"):
    q = model.layers[0].self_attn.attention_queries.save()
    pattern = model.layers[0].self_attn.attention_probabilities.save()
    out = model.layers[0].layer_output.save()
print(tuple(q.shape), tuple(pattern.shape), tuple(out.shape))
print(isinstance(q, Queries), isinstance(pattern, Pattern), isinstance(out, Pattern))
print(isinstance(q[0], Queries), isinstance(q.long(), Queries))
# ('batch', 'seq', 'hidden_size') True
# ('batch', 'num_heads', 'seq', 'qk_head_dim')
# (1, 12, 7, 64) (1, 12, 7, 7) (1, 7, 768)
# True True False
# False False
```

`isinstance(tensor, value.layout)` checks rank and dtype, not axis sizes: `q[0]` fails on rank and `q.long()` on dtype. Read the layout off the class (`Attention.attention_queries`) or off the instance's type (`type(model.layers[0].self_attn).attention_queries`); a family that redefines a value annotates it with the same name, so the layout cannot drift from the base.

## The names

The values GPT-2 serves carry these layouts, and `print(model)` shows each value's beside its description ([Names](../names.md#printmodel)).

| layout | axes | values |
| --- | --- | --- |
| `Residual` | `batch seq hidden_size` | `layer_input`, `layer_output`, `attention_output`, `mlp_output`, `token_embeddings`, the block's norm boundaries |
| `Logits` | `batch seq vocab_size` | `logits`, `lm_head_output` |
| `NextTokenProbs` | `batch vocab_size` | `next_token_probs` |
| `Tokens` | `batch seq`, integer | `input_ids`, `attention_mask` |
| `Queries` | `batch num_heads seq qk_head_dim` | `attention_queries` |
| `Keys` | `batch num_kv_heads seq qk_head_dim` | `attention_keys` |
| `Values` | `batch num_kv_heads seq head_dim` | `attention_values` |
| `Pattern` | `batch num_heads query key` | `attention_scores`, `attention_probabilities` |
| `HeadOutputs` | `batch seq num_heads head_dim` | `attention_head_outputs`, `attention_premix` |
| `Neurons` | `batch seq intermediate_size` | `mlp_activation`, `mlp_neurons` |

There are 37 names in all. The mixture's (`RouterLogits`, `ExpertWeights`, ...), the recurrent mixers' (`State`, `ScanQK`, `SSDValues`, ...) and the vision tower's (`Patches`, `ImageFeatures`) are on their Guide pages and defined beside the envoy that serves them under the Documentation tab ([components](../../documentation/components/index.md)). `nnterp.components` exports 34 of them; the root's three, `Logits`, `NextTokenProbs` and `Tokens`, are in `nnterp.standardized`. A value with no tensor annotation, such as `input_size`, has `layout` and `dims` of `None`.

## Axes named as sizes

An axis that is a fact about the model is named exactly as the size attribute that holds it (`hidden_size`, `vocab_size`, `num_heads`, `num_kv_heads`, `head_dim`, `qk_head_dim`, `intermediate_size`, `num_experts`, `top_k`), and `nnterp.axis_size(envoy, axis)` resolves it: the attribute on the envoy's class when it declares one, else on the first envoy above it that does, up to the root. The names are constants in `nnterp.components` (`HEADS` is `"num_heads"`).

```python
from nnterp import axis_size

attn = model.layers[0].self_attn
print(axis_size(attn, "num_heads"), axis_size(attn, "num_kv_heads"), axis_size(attn, "head_dim"))
print(axis_size(model.layers[0], "hidden_size"), axis_size(model, "vocab_size"))
print({axis: axis_size(attn, axis) for axis in Attention.attention_keys.dims if axis not in ("batch", "seq")})
try:
    axis_size(attn, "key")
except ValueError as error:
    print(error)
# 12 12 64
# 768 50257
# {'num_kv_heads': 12, 'qk_head_dim': 64}
# axis 'key' of a value on model.transformer.h.0.attn is not a fact about the model: no envoy from there up to the root holds a size by that name
```

`num_heads` on an attention value is that block's own count, which differs from the root's on Gemma-4, MiMo-V2-Flash and Laguna. The other axes (`batch`, `seq`, `query`, `key`, a mixer's `heads`, `key_dim`, `state_dim`, ...) are not facts about the model, so `axis_size` refuses them and their size is the tensor's.

## The sequence axis

Batch is axis 0 everywhere. The sequence axis is 1 on every value that has one, except the queries, keys and values of softmax attention, where it is 2, the `[batch, num_heads, seq, head_dim]` layout transformers hands its attention interface. `attention_head_outputs` is back to sequence first, and a pattern has two token axes, `query` then `key`. The last token is therefore `out[:, -1]` for a residual-stream value, `q[:, :, -1]` for the queries, and `pattern[:, :, -1, :]` for the last query row.

```python
print(tuple(out[:, -1].shape), tuple(q[:, :, -1].shape), tuple(pattern[:, :, -1, :].shape))
# (1, 768) (1, 12) (1, 12, 7)
```

Under grouped-query attention the keys and values are read before `repeat_kv`, so their head axis is `num_kv_heads` wide: on Qwen3-0.6B the queries come out `(1, 16, seq, 128)` and the keys `(1, 8, seq, 128)`. GPT-2 has no grouping, so the two counts coincide above.

## The exceptions

On three families one value has a layout of its own. On DeepSeek-V4 `layer_output` (and `layer_input`) is `Streams`, `[batch, seq, streams, hidden_size]`, because the block carries parallel copies of the residual stream; the contributions stay `Residual`, and code written for `Residual` runs on a `Streams` value without an error and answers per stream. On Kimi-Linear `linear_attn.decays` is `ChannelGates`, `[batch, seq, heads, key_dim]`, one decay per key channel instead of the `Gates` `[batch, seq, heads]` of the other DeltaNet families. On OPT `layer_mid` is `FlatResidual`, `[tokens, hidden_size]`, the stream with batch and sequence flattened as its block holds it around the feed-forward path.

```python
from nnterp.components import ChannelGates, FlatResidual, Streams
from nnterp.families import deepseek_v4, kimi_linear, opt

print(deepseek_v4.Layer.layer_output.layout is Streams, deepseek_v4.Layer.layer_output.dims)
print(kimi_linear.LinearAttention.decays.layout is ChannelGates, kimi_linear.LinearAttention.decays.dims)
print(opt.Layer.layer_mid.layout is FlatResidual, opt.Layer.layer_mid.dims)
# True ('batch', 'seq', 'streams', 'hidden_size')
# True ('batch', 'seq', 'heads', 'key_dim')
# True ('tokens', 'hidden_size')
```

Where a recipe must run across families, check `value.layout` (or the tensor's rank) rather than assuming `Residual`. A value of your own takes a name the same way, as its return annotation ([Custom values](../../developing/custom-values.md)).
