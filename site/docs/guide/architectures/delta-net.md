# Gated DeltaNet

On Qwen3-Next, Qwen3.5, OLMo-Hybrid and Kimi-Linear three blocks in four replace attention with a gated DeltaNet mixer, `layers[i].linear_attn`, a `LinearAttention`. It projects queries, keys and values like attention but mixes them through a recurrent state, one `[key_dim, value_dim]` matrix per head: each token decays the state by a gate, writes its key/value pair in with a strength, and the query reads against it. There is no pattern and no scores. Qwen/Qwen3.5-0.8B is the smallest public checkpoint with one.

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("Qwen/Qwen3.5-0.8B", device="cpu")
linear = [i for i, layer in enumerate(model.layers) if getattr(layer, "linear_attn", None) is not None]
softmax = [i for i, layer in enumerate(model.layers) if getattr(layer, "self_attn", None) is not None]
print(linear[:4], softmax[:4], model.config.layer_types[:4])
mix = model.layers[0].linear_attn

with model.trace("The Eiffel Tower is in the city of"):
    q = mix.attention_queries.save()
    g = mix.decays.save()
    b = mix.betas.save()
    y = mix.attention_head_outputs.save()
    state = mix.state_output.save()
    out = mix.attention_output.save()

print(tuple(q.shape), tuple(g.shape), tuple(y.shape), tuple(state.shape), tuple(out.shape))
print(g.dtype, g.max().item() <= 0, round(b.min().item(), 3), round(b.max().item(), 3))
# [0, 1, 2, 4] [3, 7, 11, 15] ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention']
# (1, 9, 16, 128) (1, 9, 16) (1, 9, 16, 128) (1, 16, 128, 128) (1, 9, 1024)
# torch.float32 True 0.044 0.996
```

A block has `self_attn` or `linear_attn`, never both, following `config.layer_types`. Decide which blocks have which outside the trace, as above, since `getattr(envoy, name, None)` inside a trace can trip a served value. `model.support(layer=i)` reports every `self_attn` value as `no self_attn module on this block` on a linear block, and every `linear_attn` value the same way on an attention block.

## The values

| value | shape | what it is |
| --- | --- | --- |
| `attention_output` | `[batch, seq, hidden_size]` | what the mixer adds to the residual stream |
| `attention_queries`, `attention_keys` | `[batch, seq, heads, key_dim]` | what the delta rule receives, before the kernel's l2-norm and the queries' `1/sqrt(key_dim)` scale |
| `attention_values` | `[batch, seq, heads, value_dim]` | the values the delta rule receives |
| `decays` | `[batch, seq, heads]`, float32 | the gate: the log of how much of the state each token keeps, non-positive |
| `betas` | `[batch, seq, heads]` | how strongly each token's pair is written, in (0, 1); in (0, 2) on OLMo-Hybrid |
| `state_input` | `[batch, heads, key_dim, value_dim]` | the state the call starts from: `None` on a fresh prompt, a copy of the cached state on a decode step |
| `state_output` | `[batch, heads, key_dim, value_dim]` | the state after the call's last token, what the next decode step starts from |
| `attention_head_outputs` | `[batch, seq, heads, value_dim]` | each head's read of the state, before the gated norm and the output projection |

`heads` is the mixer's `num_v_heads`, and the queries and keys are repeated up to it. On Kimi-Linear the decay is one per key channel, so `decays` is `[batch, seq, heads, key_dim]`. The sequence axis is 1 on every recurrent mixer, where softmax attention's queries put the heads first. Signatures are on the [`linear_attention`](../../documentation/components/linear_attention.md) and [`recurrent`](../../documentation/components/recurrent.md) pages of the Documentation tab.

## Kernels and `route_kernels`

A recurrent mixer (a gated DeltaNet, a Mamba-1 or a Mamba-2 mixer) runs its sequence through a kernel function that the modeling module calls: here `torch_chunk_gated_delta_rule` on a prompt and `torch_recurrent_gated_delta_rule` on each decode step of `generate`, chosen by the forward's own test for one token over a cached state. Everything but `attention_output` is read at that call. nnterp reads the call that fires on this step, so the same value works in a `trace` and at every step of [`tracer.iter`](https://nnsight.net/features/4_multiple_token/), and the state hands off from one step to the next:

```python
prompt = "The Eiffel Tower is in the city of"
entering, leaving = [], []                       # bound outside: a list first bound inside the body does not survive it
with model.generate(prompt, max_new_tokens=3, do_sample=False) as tracer:
    for step in tracer.iter[:3]:
        s = mix.state_input
        entering.append(s.save() if s is not None else None)
        leaving.append(mix.state_output.save())

print(entering[0] is None, torch.equal(leaving[0], entering[1]), torch.equal(leaving[1], entering[2]))
# True True True
```

The kernels must be transformers' pure-torch ones: with `flash-linear-attention` installed the names dispatch to compiled kernels with no Python source, and every kernel value reports that, naming `route_kernels` as the fix. A submodule's value (`linear_attn.in_proj_qkv.output`) and a kernel value cannot share a trace in either order, because the kernel values look up an op at the top of the mixer's forward, and the read raises `OutOfOrderError` naming it. The queries, keys and values are views that torch refuses to edit in place under autograd (`Output 0 of Select is a view and is being modified inplace`): assign them, or edit in place under `torch.no_grad()`. `attention_head_outputs`, `decays` and `betas` take in-place edits.

The chunked kernel carries the state between 64-token chunks and never materializes it per token. Until the family is routed, `state` and `states` report:

```python
print(model.support(layer=0)["linear_attn.states"])
# the state after each token is materialized only by the token-by-token kernel; the chunked kernel a prompt runs through carries it between chunks. Call nnterp.route_kernels(model.family, 'torch') before tracing this layer (slower, like attn_implementation='eager')
```

`nnterp.route_kernels(family, "torch")` rebinds the family's two kernel names, in its modeling module, to transformers' pure-torch kernels. On a gated DeltaNet both go to the token-by-token loop, and a prompt then runs through it too, slower, as eager attention is. The call is process-wide, like installing a kernel, and applies to every model of the family. `"default"` restores the import-time binding. Call it before the first trace of the layer, because a forward that nnsight has already instrumented keeps the kernel it was compiled with. `family` is the family module, or `model.family` on a loaded model.

## The state after every token

Routed, `states` is the state after every token of the call, `state` is one occurrence per token that `tracer.iter` reads or writes, and `state_after(t)` and `set_state_after(t, value)` are the same two operations as calls:

```python
import nnterp
from nnterp import StandardizedTransformer, route_kernels

route_kernels(nnterp.families.qwen3_5_text, "torch")      # before the first trace of the layer; process-wide
model = StandardizedTransformer("Qwen/Qwen3.5-0.8B", device="cpu")
mix = model.layers[0].linear_attn
prompt = "The Eiffel Tower is in the city of"

with model.trace(prompt):
    states = mix.states.save()                 # [batch, seq, heads, key_dim, value_dim]
    final = mix.state_output.save()
    clean = model.logits[0, -1].save()

with model.trace(prompt) as tracer:
    for t in tracer.iter[4]:
        mix.state = torch.zeros_like(final)    # tokens 5.. continue from an empty state
    for t in tracer.iter[5]:
        s5 = mix.state.save()
    edited = model.logits[0, -1].save()

print(tuple(states.shape), torch.equal(states[:, -1], final))
print(round(s5.norm().item(), 2), round(states[:, 5].norm().item(), 2), round((edited - clean).abs().max().item(), 2))
# (1, 9, 16, 128, 128) True
# 2.85 6.65 4.41
```

A write at token `t` changes token `t`'s own output as well, since its query reads the state it has just written. Reads in one trace follow the forward: positions before a write are read before it, positions after it afterwards, and `states`, which reads every position, goes before any write. `states` is read-only. Under `generate` the prompt is one kernel call and each decode step another over one token, so the per-token view is an inner `tracer.iter` on step 0 and `state_output` on the later steps.

!!! warning "The state is not the whole memory"
    A short convolution (width 4) in front of the kernel mixes each token's queries, keys and values with the three tokens before it. `set_state_after(t, zeros)` is not "forget everything before `t`": tokens `t + 1` to `t + 3` still see tokens up to `t` through the convolution.

[DeltaNet state](../../tutorials/delta-net-state.md) patches and tracks the state across prompts, and [Mamba](mamba.md) and [Mamba-2](mamba-2.md) serve their mixers under the same names.
