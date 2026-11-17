# A head's weights

`self_attn.value_weight(kv)` and `self_attn.output_weight(j)` are one head's slices of a block's value and output projections, and `layers[i].attention_input_gain` and `layers[i].attention_output_gain` are the gains of the norms between the residual stream and those projections. Their product is the head's OV map from the stream back into the stream. The example uses Gemma-3-270m, where the two projections' weights alone are not that map. Its norms multiply by `1 + weight`, a second norm sits between the output projection and the stream, and its four query heads share one key/value head.

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer(
    "google/gemma-3-270m", dispatch=True, device="cpu", dtype=torch.float32, attn_implementation="eager"
)
layer = model.layers[5]
attn = layer.self_attn
group = attn.num_heads // attn.num_kv_heads      # query heads per key/value head
head = 2

W_V = attn.value_weight(head // group)           # [head_dim, hidden_size]
W_O = attn.output_weight(head)                   # [hidden_size, head_dim]
g_in, g_out = layer.attention_input_gain, layer.attention_output_gain    # [hidden_size] each
with torch.no_grad():
    OV = g_out[:, None] * (W_O @ W_V) * g_in     # [hidden_size, hidden_size]

print(attn.num_heads, attn.num_kv_heads, W_V.shape, W_O.shape, OV.shape)
# 4 1 torch.Size([256, 640]) torch.Size([640, 256]) torch.Size([640, 640])
norm_weight = layer.input_layernorm.weight
print(torch.allclose(g_in, 1 + norm_weight), torch.allclose(g_in, norm_weight))
# True False
```

| accessor | shape | what it is |
| --- | --- | --- |
| `self_attn.value_weight(kv)` | `[head_dim, hidden_size]` | key/value head `kv`'s slice of the value projection, which maps what the attention reads to that head's values |
| `self_attn.output_weight(j)` | `[hidden_size, head_dim]` | head `j`'s slice of the output projection, which maps the head's output to the projection's output |
| `layers[i].attention_input_gain` | `[hidden_size]` | the gain of the norm between the stream and the attention; a vector of ones where the attention reads the stream itself, as on OLMo-2 |
| `layers[i].attention_output_gain` | `[hidden_size]` | the gain of the norm between the output projection and the stream; a vector of ones where the block adds the projection's output as it is, as on Llama, Qwen and GPT-2 |

Use the accessors for anything that multiplies a head's weights instead of reading its activations, such as an OV circuit or an attribution per head, in place of slices of `v_proj.weight` and `o_proj.weight`. Those slices are a head's weights on Llama and Qwen. GPT-2, Phi-3 and Pythia have no `v_proj`, because one projection computes the queries, keys and values there, per head on Pythia and with the weight transposed on GPT-2. A norm's gain is its `weight` on Llama and `1 + weight` on Gemma and Qwen3.5. All four work outside a trace or inside one.

`OV @ h` is the direction that head 2 adds to the stream at a position attending to a token with stream `h`, up to a positive scale, because each norm also divides a token by its own root mean square, which depends on the run. Under grouped-query attention the value projection has one slice per key/value head, and query head `j` reads key/value head `j // (num_heads // num_kv_heads)`, the `head // group` above ([Attention](attention.md#grouped-query-attention)).

## Rebuilding `attention_output`

The four factors, the stream entering the block and the pattern give back what the block adds to the stream, with the root mean squares taken from the run:

```python
with model.trace("The Eiffel Tower is in"):
    x = layer.layer_input.save()                     # the stream entering the block
    pattern = attn.attention_probabilities.save()    # [batch, num_heads, query, key]
    added = attn.attention_output.save()             # what the block adds to the stream

unit = lambda t: t / t.pow(2).mean(-1, keepdim=True).sqrt()    # an RMSNorm without its gain
with torch.no_grad():
    read = g_in * unit(x[0])                         # what the attention reads, [seq, hidden_size]
    heads = [pattern[0, h, -1] @ read @ attn.value_weight(h // group).T for h in range(attn.num_heads)]
    y = sum(attn.output_weight(h) @ heads[h] for h in range(attn.num_heads))    # the output projection's output

print(torch.allclose(g_out * unit(y), added[0, -1], atol=1e-3))
# True
```

`attn.output_weight(h) @ heads[h]` is head `h`'s share of the output projection's output at the last position. On a run, read the heads' outputs as `attention_premix`, which needs no eager attention ([Residual stream](residual-stream.md#between-the-sublayers)). [Contribution decomposition](../../tutorials/contribution-decomposition.md#per-head) takes each head's share through the final norm to a logit.

The accessors serve weights and gains only, and a bias is read off its module. Gemma's norms and projections have none. GPT-2's LayerNorm centers the stream, scales it and adds a bias, and GPT-2's projections add their own. On Qwen3.5 a gate computed by `q_proj` multiplies each head's output before the output projection, and Granite multiplies the projection's output by `residual_multiplier`. Neither step is in an accessor, so `OV` is not the head's whole map on those two families.

## Dtype and adapters

The two weights are views of the module's parameter, in its dtype and on its device. An in-place edit of one edits the model, and both require grad as the parameter does. The two gains are float32 and detached. Without `dtype=` this checkpoint loads in bfloat16, so cast the weights with `.float()` before a product with the gains. Under a PEFT adapter the two weights are the ones that the adapted projections apply.

## `nnterp.Unavailable` at the call

Load the model with `dispatch=True`, because all four accessors read the modules' weights. The accessors are methods and properties, not values, and `model.support()` does not list them. Without the weights loaded, `value_weight` and `output_weight` raise `nnterp.Unavailable` with the reason, as the read of an unavailable value does ([Availability](availability.md)):

```python
import nnterp

lazy = StandardizedTransformer("google/gemma-3-270m")
try:
    lazy.layers[5].self_attn.value_weight(0)
except nnterp.Unavailable as error:
    print(error)
# model.model.layers.5.self_attn.v_proj's weight is on the meta device: the model is not dispatched; load it with dispatch=True, or call model.dispatch(), before reading its weights
```

The two weights are unavailable in the same way on a quantized load and on a model split across GPUs ([`distributed_config=`](../loading.md#distributed_config)), and all four accessors are unavailable on `StandardizedVLLM`. `value_weight` alone is unavailable where a head's values are not one linear map of what the attention reads: under latent attention, as on DeepSeek-V3 and Kimi-K2, and on Gemma-4, which norms each head's values after the projection. `output_weight` is served there.

The signatures are on [`Attention`](../../documentation/components/attention.md#nnterp.components.attention.Attention.value_weight) and [`Layer`](../../documentation/components/layer.md#nnterp.components.layer.Layer.attention_input_gain), and the activations that these weights produce are on [Attention](attention.md).
