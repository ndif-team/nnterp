# Mamba

On Mamba and Falcon-Mamba every block is a norm and a Mamba-1 mixer, `layers[i].linear_attn`, a `SelectiveScan`. Jamba puts one on every block except the attention blocks that `attn_layer_period` places. Per channel of its inner width the mixer keeps a state of `state_dim` numbers and runs, for each token, `h = exp(dt * A) * h + dt * B * x` and `y = C . h + D * x`. nnterp serves the scan's tensors under the recurrent mixer's shared names: `C` reads the state as a query does, `B` says where a token writes as a key does, `x` is what it writes, `dt` is the write strength (`betas`) and `dt * A` the log decay (`decays`). state-spaces/mamba-130m-hf is the smallest public checkpoint.

```python
import torch
import nnterp
from nnterp import StandardizedTransformer, route_kernels

route_kernels(nnterp.families.mamba, "torch")    # before any trace: mamba_ssm's kernels have no source and need CUDA
model = StandardizedTransformer("state-spaces/mamba-130m-hf", device="cpu")
mix = model.layers[0].linear_attn

with model.trace("The Eiffel Tower is in the city of"):
    C = mix.attention_queries.save()
    B = mix.attention_keys.save()
    x = mix.attention_values.save()
    dt = mix.betas.save()
    decays = mix.decays.save()
    y = mix.attention_head_outputs.save()
    state = mix.state_output.save()
    out = mix.attention_output.save()

print(tuple(C.shape), tuple(B.shape), tuple(x.shape), tuple(dt.shape), tuple(decays.shape))
print(tuple(y.shape), tuple(state.shape), tuple(out.shape))
# (1, 10, 1, 16) (1, 10, 1, 16) (1, 10, 1536) (1, 10, 1536) (1, 10, 1536, 16)
# (1, 10, 1536) (1, 1536, 16) (1, 10, 768)
```

`route_kernels` comes first because transformers calls `mamba_ssm`'s compiled kernels when that package is installed, and they have no Python source to read inside and do not run on CPU: unrouted, a CPU trace fails inside the kernel with `Expected u.is_cuda() to be true`, even when it reads only `layer_output`. What the call does, and why a recurrent mixer's values need it, is on the [DeltaNet page](delta-net.md#kernels-and-route_kernels). On Mamba-1 the pure-torch scan is itself a token loop, and each kernel name is bound to its own pure-torch function. A Mamba block has no `self_attn` and no `mlp`: `support()` lists neither, and `layers[i].input + linear_attn.attention_output == layer_output`. Jamba's blocks are Llama's, with one mixer each.

## The values

| value | shape | what it is |
| --- | --- | --- |
| `attention_output` | `[batch, seq, hidden_size]` | what the mixer adds to the residual stream |
| `attention_queries` | `[batch, seq, 1, state_dim]` | `C`, what each token reads the state with; one group, shared by every channel |
| `attention_keys` | `[batch, seq, 1, state_dim]` | `B`, what each token writes into the state along |
| `attention_values` | `[batch, seq, channels]` | `x`, the input each channel writes, after the convolution and the activation |
| `betas` | `[batch, seq, channels]` | `softplus(dt + dt_bias)`, the step size; derived, read-only |
| `decays` | `[batch, seq, channels, state_dim]` | `dt * A`, the log decay per channel and state dimension; derived, read-only |
| `state_input` | `[batch, channels, state_dim]` | a decode step's cached state, a copy; `None` on a prompt, whose scan starts from zeros |
| `attention_head_outputs` | `[batch, seq, channels]` | `y = C . h + D * x`, before the `silu(z)` gate and `out_proj` |
| `state_output` | `[batch, channels, state_dim]` | the state after the call's last token, as the cache receives it |

`channels` is the mixer's `intermediate_size` (1536 here) and `state_dim` its `ssm_state_size` (16). The kernel takes these tensors channel-first; each value is a tokens-first view of its argument, laid back out on assignment. Unlike a DeltaNet's, `decays` has a state axis, because `A` is per channel and state dimension. `betas` and `decays` are derived from the kernel's arguments and read-only. On Falcon-Mamba and Jamba `B`, `C` and the step size pass through RMS norms before the scan, and the values are the normed ones. Signatures are on the [`selective_scan` page](../../documentation/components/selective_scan.md) of the Documentation tab.

## The state

The pure-torch scan loops over the tokens, and once routed the state after every token of a prompt exists with no second kernel: `states` stacks them, `state` is one occurrence per token for `tracer.iter` to read or write, and `state_after(t)` and `set_state_after(t, value)` are the same two operations as calls:

```python
prompt = "The Eiffel Tower is in the city of"
with model.trace(prompt):
    states = mix.states.save()                 # [batch, seq, channels, state_dim]
    final = mix.state_output.save()

with model.trace(prompt) as tracer:
    for t in tracer.iter[2]:
        mix.state = torch.zeros_like(final)    # tokens 3.. continue from an empty state
    for t in tracer.iter[3]:
        s3 = mix.state.save()

print(tuple(states.shape), torch.equal(states[:, -1], final))
print(round(s3.norm().item(), 2), round(states[:, 3].norm().item(), 2))
# (1, 10, 1536, 16) True
# 3.66 10.6
```

A write at token `t` changes token `t`'s own output too, since `y_t` reads `h_t`, and the convolution in front of the scan (width 4) still carries the three tokens before `t` into the three after it, as on a [DeltaNet](delta-net.md#the-state-after-every-token). Under `generate` the prompt runs the scan and each decode step runs `mamba_selective_state_update` over one token, the state handed from `state_output` to the next step's `state_input`:

```python
with model.generate(prompt, max_new_tokens=2, do_sample=False) as tracer:
    for step in tracer.iter[1]:                # a decode step: the state is updated first, then read
        entering = mix.state_input.save()
        leaving = mix.state_output.save()
        y1 = mix.attention_head_outputs.save()

print(tuple(entering.shape), tuple(y1.shape))
# (1, 1536, 16) (1, 1, 1536)
```

!!! warning "Read order on a decode step"
    The scan computes `y` before it returns the state, but the decode kernel updates the state first and reads `y` from it. On a decode step read `state_output` before `attention_head_outputs`, as above. The other order does not raise: `state_output` binds the next step's state, and only on the last step is the loop cut short, with a `was never reached` warning that blames the loop.

## Assign, do not edit in place

`C` and `B` are views that torch refuses to edit in place while autograd is on. Assign them, or edit in place under `torch.no_grad()`. `attention_values` and `attention_head_outputs` take in-place edits either way, and a write to `state_output` changes what the next decode step starts from, not this call's output.

```python
with model.trace(prompt):
    mix.attention_queries = mix.attention_queries * 0        # assign: C is a view of the kernel's argument
    y_noC = mix.attention_head_outputs.save()

with model.trace(prompt):
    with torch.no_grad():
        mix.attention_queries[:, -1] = 0                     # or edit in place without autograd
    y_noC_last = mix.attention_head_outputs.save()

print(torch.allclose(y_noC[:, -1], y_noC_last[:, -1]))

try:
    with model.trace(prompt):
        mix.attention_queries[:, -1] = 0
except RuntimeError as e:
    print(str(e)[:60])
try:
    with model.trace(prompt):
        mix.betas = mix.betas * 0
except AttributeError as e:
    print(e)
# True
# Output 0 of Select is a view and is being modified inplace.
# betas is derived and read-only
```

Mamba-2's mixer has the same names and a different state: [Mamba-2](mamba-2.md).
