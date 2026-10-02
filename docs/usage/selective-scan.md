---
title: Mamba-1 Selective Scan
one_liner: The `linear_attn` values on Mamba, Falcon-Mamba and Jamba's Mamba blocks, read at the selective-scan kernel call as tokens-first views, and the per-token state of the scan's own token loop.
tags: [usage, state-space, mamba, selective-scan, state, mamba, falcon_mamba, jamba]
related: [docs/usage/delta-net.md, docs/usage/vocabulary.md, docs/usage/availability.md, docs/usage/layouts.md, docs/usage/generation.md, docs/developing/recurrent-mixer-internals.md]
sources: [nnter/components/selective_scan.py, nnter/components/recurrent.py, nnter/components/eproperty.py, nnter/families/mamba.py, nnter/families/falcon_mamba.py, nnter/families/jamba.py, tests/families/scan_suite.py, tests/families/test_mamba.py]
---

# Mamba-1 Selective Scan

## What this is for

Mamba and Falcon-Mamba are pure state-space models: every block is a norm
and a Mamba-1 mixer, with no attention and no MLP. Jamba mixes such blocks
with attention blocks and a mixture of experts. The mixer, `layers[i].linear_attn`
(native `mixer`, or `mamba` on Jamba), keeps a state of `state_dim` numbers
per channel of its inner width and runs, per channel, for each token:

```
h_t = exp(dt_t * A) * h_{t-1} + dt_t * B_t * x_t
y_t = C_t . h_t + D * x_t
```

`dt` is a per-token, per-channel step size, `B` and `C` are per-token vectors
of `state_dim` numbers shared by every channel, `A` a learned
`[channels, state_dim]` matrix of negative rates. The output is `y` gated by
`silu(z)` and projected by `out_proj`. `nnter.components.SelectiveScan` gives
the mixer the names a gated DeltaNet has where they mean the same thing: `C`
reads the state as a query does, `B` writes into it as a key does, `x` is what
is written, `dt` is how strongly (`betas`), `dt * A` how much of the state
each token keeps (`decays`). It is a `RecurrentMixer`, like `LinearAttention`
([delta-net.md](delta-net.md)), so the kernel switch and the per-token state
are the base's.

> **How this mixer differs from a gated DeltaNet and Mamba-2.** The state is per
> *channel*, `[channels, state_dim]`, value side first (`ScanState`), not per head. The
> update is a plain decay-and-add, with no delta-rule correction, and the decay
> `decays` has a `state_dim` axis (`A` is per channel and state dimension). `betas` and
> `decays` are derived from the kernel's arguments and read-only. The per-token state is
> readable and writable (`state`, `set_state_after`) once routed, because the pure-torch
> scan is itself a token loop. A short convolution before the scan (width 4) also
> carries the last few tokens. [vocabulary.md](vocabulary.md#same-name-different-meaning)
> puts the three mixers side by side.

## Canonical pattern

```python
import torch
import nnter
from nnter import StandardizedTransformer, route_kernels

route_kernels(nnter.families.mamba, "torch")    # before the first trace; see "The kernels"
model = StandardizedTransformer("state-spaces/mamba-130m-hf", dispatch=True)
mix = model.layers[0].linear_attn
prompt = "The Eiffel Tower is in the city of"

with model.trace(prompt):
    x = mix.attention_values.save()           # [batch, seq, channels]
    b = mix.betas.save()                      # [batch, seq, channels]: the step size dt, > 0
    y = mix.attention_head_outputs.save()     # [batch, seq, channels]: C.h + D x, before the gate
    state = mix.state_output.save()           # [batch, channels, state_dim]: after the last token
    out = mix.attention_output.save()         # [batch, seq, hidden]: what the mixer adds to the stream
```

A Mamba block's only contribution is the mixer's:
`layers[i].input + linear_attn.attention_output == layer_output`, and
`support()` lists no `self_attn` and no `mlp` values. On Jamba the blocks are
Llama's (`input + attention_output + mlp_output == layer_output`), with
`self_attn` on one block in `attn_layer_period` and `linear_attn` on the rest;
decide which is which outside the trace, as on a DeltaNet hybrid.

## The kernels

The forward calls `mamba_selective_scan(x, dt, A, B, C, D=, z=, delta_bias=, ...)`
on a prompt and `mamba_selective_state_update(state, x, dt, A, B, C, D, z=,
dt_bias=, ...)` on each decode step, and every value but `attention_output` is
read at that call. transformers binds both names to `mamba_ssm`'s compiled
CUDA kernels when it is installed: they have no Python source to read inside,
and they do not run on CPU. `route_kernels(family, "torch")` binds each name
to its own pure-torch function, process-wide:

- The pure-torch selective scan *is* a token loop, so after routing a prompt's
  state after every token exists without a second kernel, and `state`,
  `states`, `state_after` and `set_state_after` work (the DeltaNet reroutes
  the prompt through its decode kernel for the same result).
- Unrouted with `mamba_ssm` installed, every kernel value reports `read inside
  transformers' pure-torch mamba_selective_scan, but this process dispatches
  it to an optimized kernel (mamba_ssm) with no Python source; uninstall it, or
  call nnter.route_kernels(model.family, 'torch'), to read these`.
- Route before the layer's forward is traced; `route_kernels(family,
  "default")` restores the module's own bindings. `family` is the family
  module (`nnter.families.mamba`, `falcon_mamba`, `jamba`) or `model.family`.

## The values

Each is a view of the kernel's argument laid out tokens first: the kernel's
tensors are channel-first (`x` and `dt` `[batch, channels, seq]`, `B` and `C`
`[batch, state_dim, seq]`; a decode step has no sequence axis), so a read
transposes, an assignment is laid back out, and an in-place edit
(`mix.attention_values[:, -1] = 0`) reaches the kernel. `attention_queries` and
`attention_keys` (`C`, `B`) are views torch refuses to edit in place while autograd is
on (`RuntimeError: Output 0 of Select is a view and is being modified inplace`):
assign them, or edit in place under `torch.no_grad()`.

| value | what it is | layout |
| --- | --- | --- |
| `attention_output` | what the mixer adds to the residual stream | `Residual`: `batch seq hidden` |
| `attention_queries` | `C`, what each token reads the state with; one group, shared by every channel | `ScanQK`: `batch seq groups state_dim` |
| `attention_keys` | `B`, what each token writes into the state along | `ScanQK`: `batch seq groups state_dim` |
| `attention_values` | `x`, the input each channel writes, after the conv and the activation | `ScanValues`: `batch seq channels` |
| `betas` | `softplus(dt + dt_bias)`, the step size; derived, read-only | `ScanSteps`: `batch seq channels` |
| `decays` | `dt * A`, the log of how much of the state each token keeps, per channel *and* state dimension; derived, read-only | `ScanDecays`: `batch seq channels state_dim` |
| `state_input` | a decode step's cached state (a copy; assign to replace it); `None` on a prompt, whose scan starts from zeros | `ScanState`: `batch channels state_dim` |
| `attention_head_outputs` | `y = C . h + D x`, before the `silu(z)` gate and `out_proj`; a binding inside the kernel | `ScanValues`: `batch seq channels` |
| `state_output` | the state after the call's last token, as the cache receives it | `ScanState`: `batch channels state_dim` |
| `state`, `states` | the state after one token, after every token | `ScanState`, `ScanStates`: `batch seq channels state_dim` |

`channels` is the mixer's `intermediate_size` (`expand * hidden_size`),
`state_dim` its `ssm_state_size`, `groups` is 1. On Falcon-Mamba and Jamba,
`B`, `C` and the step-size projection pass through RMS norms before the scan:
the values are the normed ones the scan uses. `decays` differs from a
DeltaNet's (one number per head) in having a `state_dim` axis: `A` is
per channel and state dimension.

A write to `attention_queries`, `attention_keys`, `attention_values` or
`attention_head_outputs` replaces what the scan receives or returns. A write
to `state_output` changes what the next decode step starts from (the cache),
not this call's output. `betas`, `decays` and `states` are read-only, and
assigning `state_input` on a prompt raises `ValueError: a prompt's selective
scan starts from zeros and takes no state to replace`.

## Under `generate`

The values follow whichever kernel the forward calls: the scan on the
prompt, the single-step update on each cached one-token call (the forward's
own condition, `use_precomputed_states and seq_len == 1`). The state hands off
from step to step:

```python
entering, leaving = [], []                       # outside the block: a name bound inside does not survive it
with model.generate(prompt, max_new_tokens=3, do_sample=False) as tracer:
    for step in tracer.iter[:3]:
        s = mix.state_input
        entering.append(s.save() if s is not None else None)
        leaving.append(mix.state_output.save())

entering[0] is None                              # the prompt's scan starts from zeros
torch.equal(leaving[0], entering[1])             # step 1 starts from what step 0 left
```

## The state after every token

`state` is one occurrence per token of the scan's loop, so nnsight's own
iteration walks it, and an assignment there is a write the following tokens
continue from; `states` stacks every position of the call; `state_after(t)`
and `set_state_after(t, value)` are the same as calls. They work as on a
DeltaNet ([delta-net.md](delta-net.md), "The state after every token"):

```python
with model.trace(prompt):
    states = mix.states.save()                 # [batch, seq, channels, state_dim]
    final = mix.state_output.save()            # == states[:, -1]

with model.trace(prompt) as tracer:
    for t in tracer.iter[1]:
        mix.state = torch.zeros_like(final)    # a write at token 1: tokens 2.. continue from zeros
    for t in tracer.iter[2]:
        s2 = mix.state.save()
```

On a decode step the per-token state is the single update's, one token:
`states` is `[batch, 1, channels, state_dim]`.

A write of the state after token `t` changes token `t`'s own output as well (`y_t` reads
`h_t`), and it is not everything the block carries forward: the short convolution
before the scan mixes each token's input with the three before it. On mamba-130m,
zeroing every block's state after token 4 of a ten-token prompt and running the last
five tokens alone give logits that differ by up to 87 and different top tokens at three
of the five positions; `set_state_after(t, zeros)` is not a clean "forget everything
before `t`".

## Gotchas

- **Route before the first trace.** With `mamba_ssm` installed nothing runs on
  CPU and nothing inside the kernels is readable until `route_kernels(family,
  "torch")`.
- **Read order differs between the kernels.** In the scan, `y`
  (`attention_head_outputs`) is computed before the returned state
  (`state_output`); in the decode step the state is updated first and `y`
  read from it. Under `generate`, read `attention_head_outputs` before
  `state_output` on step 0 and after it on later steps, or read them in
  separate runs. The wrong order on a decode step does not raise: `state_output`
  silently binds the *next* step's state (on mamba-130m, step 1's read returns step 2's
  state), and only on the last step is the loop cut short, with nnsight's `was never
  reached` warning.
- **Comparing with transformers' `output_hidden_states`**: a Mamba model's
  `hidden_states` has no embedding entry. `hidden_states[i]` is block `i`'s
  `layer_output` (not `hidden_states[i + 1]`, as on a transformer), and the last entry is
  the stream after `norm_f`.
- **In-place edits on `attention_queries` and `attention_keys` need `torch.no_grad()`**;
  assignment always works.
- **`use_mambapy`.** A checkpoint loaded with `use_mambapy=True`, with
  `mambapy` installed, runs a parallel scan that binds no per-token state:
  `state` and `states` say so.
- **`layer_output` is float32** with `residual_in_fp32` (Mamba, Falcon-Mamba),
  whatever the load dtype; the model casts the stream to `lm_head`'s dtype and
  returns float32 logits, and `project_on_vocab` does the same.
- **No head sizes.** A pure Mamba config has no attention heads:
  `model.num_heads` and `model.head_dim` have nothing to read.
  `model.intermediate_size` is the mixer's inner width.

## Related

- [delta-net.md](delta-net.md), the other `RecurrentMixer`, and the per-token state in full.
- [layouts.md](layouts.md), every value's named layout.
- [generation.md](generation.md), values per decode step.
- [../developing/recurrent-mixer-internals.md](../developing/recurrent-mixer-internals.md), how the kernel choice, the views and the per-token state are built.
