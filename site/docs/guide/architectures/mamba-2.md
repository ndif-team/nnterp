# Mamba-2

On Mamba-2, Nemotron-H, Bamba and Falcon-H1 the recurrent mixer `layers[i].linear_attn` is a `StateSpace`, a Mamba-2 (SSD) mixer: linear attention with one scalar decay per head. Per head, with the state a `head_dim` by `state_dim` matrix, each token does `h = exp(dt * A) * h + dt * x B^T` and `y = h C + D * x`. The shared names mean what they mean on a [gated DeltaNet](delta-net.md): `C` is `attention_queries`, `B` is `attention_keys`, `x` is `attention_values`, `dt` is `betas`, `A * dt` is `decays` and `y` is `attention_head_outputs`. AntonV/mamba2-130m-hf is the smallest public checkpoint. On it every block is the mixer alone. Nemotron-H and Bamba mix such blocks with attention and MLP blocks, and Falcon-H1 runs a mixer and attention side by side on every block.

```python
import torch
import nnterp
from nnterp import StandardizedTransformer, route_kernels

route_kernels(nnterp.families.mamba2, "torch")   # before any trace: mamba_ssm's kernels have no source and need CUDA
model = StandardizedTransformer("AntonV/mamba2-130m-hf", device="cpu")
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
print(decays.dtype, decays.max().item() <= 0, dt.min().item() > 0)
# (1, 10, 1, 128) (1, 10, 1, 128) (1, 10, 24, 64) (1, 10, 24) (1, 10, 24)
# (1, 10, 24, 64) (1, 24, 128, 64) (1, 10, 768)
# torch.float32 True True
```

`route_kernels` comes first, as on the [DeltaNet page](delta-net.md#kernels-and-route_kernels): with `mamba_ssm` installed transformers calls its compiled kernels, which have no source to read and do not run on CPU. Unrouted, a CPU trace fails inside the kernel (`Pointer argument cannot be accessed from Triton` when a GPU is visible). On Mamba-2 only the two scan kernels are rerouted, and a prompt keeps the chunked scan.

## The values

| value | SSD | shape | what it is |
| --- | --- | --- | --- |
| `attention_output` | | `[batch, seq, hidden_size]` | what the mixer adds to the residual stream |
| `attention_queries` | `C` | `[batch, seq, groups, state_dim]` | what reads the state; one per group of heads |
| `attention_keys` | `B` | `[batch, seq, groups, state_dim]` | where each token writes into the state; one per group |
| `attention_values` | `x` | `[batch, seq, heads, head_dim]` | what each token writes |
| `betas` | `dt` | `[batch, seq, heads]` | `softplus(dt + dt_bias)`, clamped to `time_step_limit` on a prompt: the write strength |
| `decays` | `A * dt` | `[batch, seq, heads]`, float32 | the log of how much of the state each token keeps, non-positive |
| `state_input` | `h` in | `[batch, heads, state_dim, head_dim]` | the state the call starts from: `None` on a fresh prompt, a copy of the cached state on a decode step |
| `state_output` | `h` out | `[batch, heads, state_dim, head_dim]` | the state after the call's last token |
| `attention_head_outputs` | `y` | `[batch, seq, heads, head_dim]` | each head's read of the state plus the `D` skip, before the gated norm and `out_proj` |

`groups` is the mixer's `n_groups`, 1 here (the heads of a group share `B` and `C`), `heads` its `num_heads` (24), `head_dim` 64 and `state_dim` its `ssm_state_size` (128). The state is served key side first, the layout every recurrent mixer's state shares and the transpose of the cache's buffer. An assignment is transposed back. Assign `attention_queries`, `attention_keys` and `attention_values` rather than editing them in place, as on the other recurrent mixers. Signatures are on the [`state_space` page](../../documentation/components/state_space.md) of the Documentation tab.

## `betas` and `decays` are one argument

Both are the kernel's `dt` seen through `dt_bias`, the softplus and `A`. An assignment is carried back into `dt`, and the kernel computes the gate that was written. An in-place edit (`mix.betas[:, t] = 0`) does not reach `dt`.

```python
prompt = "The Eiffel Tower is in the city of"
with model.trace(prompt):
    mix.decays = mix.decays * 0.5                 # half the log decay is half of dt: half-strength writes too
    halved = mix.betas.save()

with model.trace(prompt):
    betas = mix.betas.clone()
    betas[:, 3] = 0                               # token 3 writes nothing and decays nothing
    mix.betas = betas
    skipped = model.logits[0, -1].save()

with model.trace(prompt):
    mix.betas = mix.betas                         # the round trip through the softplus's inverse rounds dt
    written_back = model.logits[0, -1].save()

with model.trace(prompt):
    clean = model.logits[0, -1].save()

print(torch.allclose(halved, dt * 0.5, atol=1e-6))
print((skipped - clean).abs().max().item() > 0, (written_back - clean).abs().max().item())
# True
# True 0.000125885009765625
```

!!! warning "Writing `decays` rewrites `betas`"
    There is no way to change the decay alone. `decays * 0.5` halves `dt`: each token keeps more of the state and writes at half strength. `decays = 0` ("keep everything") sets `dt` to zero, which writes nothing, and the state stays at zero through the whole prompt.

`betas[:, t] = 0` is the exact "skip this token": no write and `exp(0) = 1`, no decay, and the state after token `t` equals the state after `t - 1`. Nemotron-H's scan clamps `dt` to `time_step_min` first, and there the state still moves. A written `betas` must be positive or zero, and on a prompt the kernel clamps it to `time_step_limit` again. Writing the same values back is not a no-op: the round trip rounds `dt`, by 1.3e-4 in the last logit here. Compare an edit against an unedited run, not against a write-back.

## The state after every token

The chunk scan computes the state at every chunk boundary in one tensor, and between boundaries there is nothing to read:

```python
from nnterp import chunk_per_token

print(model.support(layer=0)["linear_attn.states"])
chunk_per_token(model)                            # this model only; undo with chunk_per_token(model, False)
with model.trace(prompt):
    states = mix.states.save()                    # [batch, seq, heads, state_dim, head_dim]
    final = mix.state_output.save()

with model.trace(prompt):
    betas = mix.betas.clone()
    betas[:, 3] = 0
    mix.betas = betas
    held = mix.states.save()

print(tuple(states.shape), torch.equal(states[:, -1], final), torch.equal(held[:, 3], held[:, 2]))
print(model.support(layer=0)["linear_attn.set_state_after"])
chunk_per_token(model, False)
# the chunk scan materializes the state only at chunk boundaries, every 256 tokens (chunk_size=256); call nnterp.chunk_per_token(model) to set every mixer's chunk_size to 1, so every token is a boundary (slower on long prompts)
# (1, 10, 24, 128, 64) True True
# the chunk scan computes every boundary state in one cumulative step from the initial state, so a state written at token t does not flow into later tokens' states; assign state_input to change where a call starts
```

`nnterp.chunk_per_token(model)` sets every Mamba-2 mixer's `chunk_size` to 1. Every token is then a boundary, and `states` and `state_after(t)` read the state after every token of the call. The call touches this model only, because the forward reads the attribute on every call, where `route_kernels` rebinds a family's kernels for the process. `chunk_per_token(model, False)` restores the chunk size each mixer was built with. The recurrence between chunks is quadratic in their number, and a long prompt runs markedly slower. The same scan in another order also moves the logits by rounding. `states` is a read-only copy. `state` and `set_state_after` stay unavailable either way, with the reasons `support()` prints: the scan has no per-token occurrence for `tracer.iter` to walk, as a DeltaNet has, and it computes every boundary state in one cumulative step from the initial state, in which a state written at one token does not flow into the next. To change where a call starts, assign `state_input`.

## The state between steps

A prompt runs `mamba2_chunk_scan` and each decode step runs `mamba2_selective_state_update`, which takes its arguments in other places and has no sequence axis. The values select from the kernel that fires, and a decode step's tensors are served with a sequence axis of 1.

```python
entering, leaving = [], []
with model.generate(prompt, max_new_tokens=3, do_sample=False) as tracer:
    for step in tracer.iter[:3]:
        s = mix.state_input
        entering.append(s.save() if s is not None else None)
        leaving.append(mix.state_output.save())

print(entering[0] is None, torch.equal(leaving[0], entering[1]), torch.equal(leaving[1], entering[2]))
# True True True
```

The update writes the new state into the cache and returns only `y`, which is where a decode step's `state_output` is read. Read it before `attention_head_outputs` on a decode step, the trap the [Mamba page](mamba.md#the-state) describes. On Falcon-H1-0.5B and the Falcon-H1-Tiny checkpoints, which set `mamba_rms_norm` off, a decode step's `attention_head_outputs` is gated by `silu(z)` where a prompt's is not.
