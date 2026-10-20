# Residual stream

`layer_output` is the residual stream leaving block `i`, `layer_input` the stream entering it, and `attention_output` and `mlp_output` what the two sublayers add to it. All four are `[batch, seq, hidden_size]` tensors on every family, and the two contributions are defined by one identity: `layer_input + attention_output + mlp_output == layer_output`.

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2")

with model.trace("The Eiffel Tower is in"):
    x = model.layers[5].layer_input.save()
    attn = model.layers[5].self_attn.attention_output.save()
    mlp = model.layers[5].mlp.mlp_output.save()
    out = model.layers[5].layer_output.save()

print(x.shape, attn.shape, mlp.shape, out.shape)
# torch.Size([1, 7, 768]) torch.Size([1, 7, 768]) torch.Size([1, 7, 768]) torch.Size([1, 7, 768])
print(torch.equal(x + attn + mlp, out))
# True
```

The identity is exact on a sequential block. A parallel block, such as Pythia's or Falcon's, sums the three terms in another order and holds it to rounding (one float16 ulp on Pythia-70m), so compare with a tolerance in the block's dtype there.

## A tensor on every family

A block's native `.output` is a tuple on GPT-J, GPT-Neo, BLOOM, MPT and Falcon. `layer_output` is the tensor the block returned, whatever wraps it:

```python
bloom = StandardizedTransformer("bigscience/bloom-560m")

with bloom.trace("The Eiffel Tower is in"):
    raw = bloom.layers[2].output.save()
    out = bloom.layers[2].layer_output.save()

print(type(raw), type(out), torch.equal(raw[0], out))
# <class 'tuple'> <class 'torch.Tensor'> True
```

Assigning `layer_output` on such a block replaces the first element and leaves the others as they were, so the next block still receives a tuple. `attention_output` does the same where the attention module returns `(attn_output, attn_weights)`.

A contribution is what the block adds to the stream, which is not always the module's return. On the sandwich-norm families, such as Gemma-2 and OLMo-2, `attention_output` is the post-attention norm's output and `self_attn.output[0]` is the tensor before that norm. On BLOOM and MPT the module adds the residual inside, and the contribution is the tensor entering that add. The identity is what to rely on, not `self_attn.output`.

## Between the sublayers

Between `layer_input` and `layer_output` the block has four more module boundaries, and each sublayer two more inside it:

| value | where it is read | shape |
| --- | --- | --- |
| `layers[i].attention_norm_output` | `input_layernorm.output`, what the attention reads | `[batch, seq, hidden_size]` |
| `layers[i].self_attn.attention_premix` | `o_proj.input`, every head's output before the output projection | `[batch, seq, num_heads, head_dim]` |
| `layers[i].layer_mid` | `post_attention_layernorm.input`, the stream after the attention add | `[batch, seq, hidden_size]` |
| `layers[i].mlp_norm_output` | `post_attention_layernorm.output` | `[batch, seq, hidden_size]` |
| `layers[i].mlp_input` | `mlp.input`, what the MLP is called with | `[batch, seq, hidden_size]` |
| `layers[i].mlp.mlp_activation` | `act_fn.output` | `[batch, seq, intermediate_size]` |
| `layers[i].mlp.mlp_neurons` | `down_proj.input`, what the down projection sums | `[batch, seq, intermediate_size]` |

```python
block = model.layers[5]
with model.trace("The Eiffel Tower is in"):
    x = block.layer_input.save()
    normed = block.attention_norm_output.save()
    premix = block.self_attn.attention_premix.save()
    attn = block.self_attn.attention_output.save()
    mid = block.layer_mid.save()
    mlp_normed = block.mlp_norm_output.save()
    mlp_in = block.mlp_input.save()
    act = block.mlp.mlp_activation.save()
    neurons = block.mlp.mlp_neurons.save()

print(premix.shape, act.shape, neurons.shape)
# torch.Size([1, 7, 12, 64]) torch.Size([1, 7, 3072]) torch.Size([1, 7, 3072])
print(torch.equal(mid, x + attn), torch.equal(mlp_normed, mlp_in), torch.equal(act, neurons))
# True True True
```

`layer_mid` is `layer_input + attention_output`, and `mlp_norm_output` is `mlp_input` where nothing sits between the norm and the MLP. On GPT-2's plain MLP `mlp_neurons` is `mlp_activation` itself; on a gated MLP (Llama, Qwen) it is `act(gate_proj(x)) * up_proj(x)`, a different tensor at the same place. The keys above are those of a pre-norm block, under the standard names of [Names](../names.md). A block built otherwise points a value at its own module or marks it unavailable with the reason: a parallel block has no `layer_mid`, a Gemma-2 block norms the MLP's input in `pre_feedforward_layernorm`, and a routed mixture of experts has no single `mlp_activation` or `mlp_neurons` ([Availability](availability.md)).

## Reading, editing in place, assigning

Every value on this page is the live tensor, so an in-place edit reaches the model and an assignment replaces it, as with [`.output`](https://nnsight.net/features/2_setting/):

```python
with model.trace("The Eiffel Tower is in"):
    clean = model.logits.save()

with model.trace("The Eiffel Tower is in"):
    model.layers[5].self_attn.attention_output[:, -1] = 0
    model.layers[5].mlp.mlp_output[:] = 0
    model.layers[7].layer_output = model.layers[7].layer_output * 2
    edited = model.logits.save()

print((edited - clean).abs().max().item())
# 31.438919067382812
```

Within one trace, read a block's values in forward order. `model.order(i)` gives that order for block `i`, with equal ranks for values read at one location, and reading `layer_input` after `mlp_output` raises nnsight's `OutOfOrderError`:

```python
print(model.order(5))
# {'layer_input': 0, 'attention_norm_output': 1, 'self_attn.attention_premix': 2, 'self_attn.attention_output': 3, 'layer_mid': 4, 'mlp_norm_output': 5, 'mlp_input': 6, 'mlp.mlp_activation': 7, 'mlp.mlp_neurons': 8, 'mlp.mlp_output': 9, 'layer_output': 10}
```

The identity takes another form on Gemma-4, which multiplies the sum by a per-block `layer_scalar`, on Doge and ZAYA, which gate the stream per channel before each add, and on DeepSeek-V4, where `layer_output` is `hc_mult` parallel streams, `[batch, seq, streams, hidden_size]`. Each family's page in the [Documentation](../../documentation/families/index.md) tab gives its formula, and [`Layer`](../../documentation/components/layer.md) the signatures.

The values inside the attention sublayer are on the [Attention](attention.md) page.
