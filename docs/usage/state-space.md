---
title: Mamba-2 State-Space Mixers
one_liner: The `linear_attn` values on Mamba-2 (SSD) blocks — Mamba-2, Nemotron-H, Bamba, Falcon-H1 — what C, B, x and dt are called, the two kernels a prompt and a decode step run, the state handed between steps, and the state after every token with `chunk_per_token`.
tags: [usage, hybrid, state-space, mamba2, ssd, state, nemotron_h, bamba, falcon_h1]
related: [docs/usage/delta-net.md, docs/usage/vocabulary.md, docs/usage/availability.md, docs/usage/layouts.md, docs/usage/generation.md, docs/developing/recurrent-mixer-internals.md]
sources: [nnterp/components/state_space.py, nnterp/components/eproperty.py, nnterp/components/recurrent.py, nnterp/components/layer.py, nnterp/families/mamba2.py, nnterp/families/nemotron_h.py, nnterp/families/bamba.py, nnterp/families/falcon_h1.py, tests/families/ssd.py, tests/families/test_mamba2.py, tests/families/test_nemotron_h.py]
---

# Mamba-2 State-Space Mixers

## What this is for

Mamba-2, Nemotron-H (Nemotron-3 Nano and Super), Bamba and Falcon-H1 mix the
sequence in some or all blocks with a Mamba-2 (SSD) mixer, at
`layers[i].linear_attn`, the recurrent mixer's standard name on every hybrid.
SSD is linear attention with one scalar decay per head. Per head, with the
state `h` a `head_dim` by `state_dim` matrix, each token does

```
h = exp(dt * A) * h + dt * x B^T        # decay the state, write the token's value x at key B
y = h C + D * x                         # read it with query C, plus a skip
```

so `nnterp.components.StateSpace` gives the mixer the names a gated DeltaNet
(`LinearAttention`, [delta-net.md](delta-net.md)) has: `C` is
`attention_queries`, `B` is `attention_keys`, `x` is `attention_values`, `dt`
is `betas` (the write strength), `A * dt` is `decays` (the log decay), `y` is
`attention_head_outputs`, and the state entering and leaving the layer is
`state_input` / `state_output`. It is a `RecurrentMixer`, so the kernel switch
and the routing below are the base's.

> **How this mixer differs from a gated DeltaNet and Mamba-1.** The state is served
> `[state_dim, head_dim]` per head (`B`'s side first), the transpose of the cache. The
> update is a plain decay-and-add, with no delta-rule correction. `betas` and `decays` are
> two views of the kernel's one `dt` argument, not two gates: writing either rewrites
> both. The state after every token is readable (`states`, after `chunk_per_token`) but
> not writable per token: `state` and `set_state_after` do not exist, and a write goes
> through `state_input`. A short convolution before the scan (width 4) also carries the
> last few tokens. [vocabulary.md](vocabulary.md#same-name-different-meaning) puts the
> three mixers side by side.

## Canonical pattern

```python
import torch
import nnterp
from nnterp import StandardizedTransformer, route_kernels

route_kernels(nnterp.families.nemotron_h, "torch")     # when mamba_ssm is installed: before the first trace
model = StandardizedTransformer("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", attn_implementation="eager")
prompt = "The Eiffel Tower is in the city of"

ssm = [i for i, layer in enumerate(model.layers) if getattr(layer, "linear_attn", None) is not None]
mix = model.layers[ssm[0]].linear_attn

with model.trace(prompt):
    C = mix.attention_queries.save()        # [batch, seq, groups, state_dim]
    B = mix.attention_keys.save()           # [batch, seq, groups, state_dim]
    x = mix.attention_values.save()         # [batch, seq, heads, head_dim]
    dt = mix.betas.save()                   # [batch, seq, heads], > 0
    log_decay = mix.decays.save()           # [batch, seq, heads], float32, <= 0
    y = mix.attention_head_outputs.save()   # [batch, seq, heads, head_dim]
    state = mix.state_output.save()         # [batch, heads, state_dim, head_dim]: after the last token
    out = mix.attention_output.save()       # [batch, seq, hidden]: what the mixer adds to the stream
```

Decide which blocks hold a Mamba-2 mixer *outside* the trace, as above.

## Which blocks, and what `support()` says

| family | blocks | `linear_attn` is | the rest of the block |
| --- | --- | --- | --- |
| `mamba2` | every block | `mixer` | the pre-norm keeps its native name `norm` (no `input_layernorm` alias); no `self_attn`, no `mlp` |
| `nemotron_h` | `layers_block_type == "linear_attention"` | `mixer` | one sublayer per block: the `mixer` is `linear_attn`, `self_attn` or `mlp` by its class |
| `bamba` | all but `attn_layer_indices` | `mamba` | `self_attn` on the others; `feed_forward` as `mlp` on every block |
| `falcon_h1` | every block, beside `self_attn` | `mamba` | both mixers in parallel, then `feed_forward` as `mlp` |

On Mamba-2 the block is the mixer alone, so `support()` lists no `self_attn.*`
and no `mlp.*` value, and `layers[i].input + linear_attn.attention_output ==
layer_output`. On Nemotron-H each block has one of `linear_attn`, `self_attn`
and `mlp`, and `support()` reports the other two `no ... module on this block`
per block; the identity is the input plus that one sublayer's contribution.
On Falcon-H1 it has four terms, both mixers and the MLP, each mixer's
contribution scaled by its µP multiplier ([families.md](../reference/families.md)).

`states` and `state_after` need `nnterp.chunk_per_token(model)` (below);
`state` and `set_state_after` are unavailable on every Mamba-2 mixer, with the
reasons in [The state after every token](#the-state-after-every-token).

## The values

| value | SSD | what it is | layout |
| --- | --- | --- | --- |
| `attention_output` | | what the mixer adds to the residual stream | `Residual`: `batch seq hidden` |
| `attention_queries` | `C` | what reads the state, after the conv and the activation; one per group of heads | `SSDQueries`: `batch seq groups state_dim` |
| `attention_keys` | `B` | where each token writes into the state; one per group of heads | `SSDKeys`: `batch seq groups state_dim` |
| `attention_values` | `x` | what each token writes | `SSDValues`: `batch seq heads head_dim` |
| `betas` | `dt` | `softplus(dt + dt_bias)` (clamped to `time_step_limit` on a prompt): the write strength | `Gates`: `batch seq heads` |
| `decays` | `A * dt` | the log of how much of the state each token keeps; float32, non-positive | `Gates`: `batch seq heads` |
| `state_input` | `h` in | the state the call starts from: `None` on a fresh prompt, the cached state on a decode step (a copy) | `State`: `batch heads key_dim value_dim` |
| `state_output` | `h` out | the state after the call's last token: what the next decode step starts from | `State`: `batch heads key_dim value_dim` |
| `attention_head_outputs` | `y` | each head's read of the state plus the `D` skip, before the gated norm and `out_proj` | `SSDHeadOutputs`: `batch seq heads head_dim` |
| `states` | `h` per token | the state after every token of the call; needs `chunk_per_token`; read-only, a copy | `States`: `batch seq heads key_dim value_dim` |

`groups` is the mixer's `n_groups` (the heads in a group share `B` and `C`),
`heads` its `num_heads`, `head_dim` its `head_dim` and `state_dim` its
`ssm_state_size`. The state is served key side first, `[batch, heads,
state_dim, head_dim]`, the shared `State` layout (`key_dim` is `state_dim`,
`value_dim` is `head_dim`): the transpose of the cache's `[batch, heads,
head_dim, state_dim]`. An assignment is transposed back.

Everything but `attention_output` is read at the scan kernel call. Assign
`attention_queries`, `attention_keys`, `attention_values`,
`attention_head_outputs`, `state_input` or `state_output` to replace them, or
edit the first four in place (`mix.attention_head_outputs[:, -1] = 0` reaches
the model). The queries, keys and values are views torch refuses to edit in place
while autograd is on (`RuntimeError: Output 0 of Select is a view and is being
modified inplace`): assign them, or edit in place under `torch.no_grad()`.
`state_input` is a clone: the decode kernel updates the cache's buffer in place.

The values of one call can be read together in one trace in forward order
(`attention_queries`, `betas`, `attention_head_outputs`, `state_output`, ...):
the kernel's arguments are read from the model once per call, and every value
that needs one of them (`betas` needs `dt_bias`, a prompt's `state_output`
whether the scan returns a state) takes it from that one read.

### `betas` and `decays` are one argument

Both are the kernel's `dt` argument seen through `dt_bias`, the softplus and
`A`: `betas` is the step `dt` itself, the write strength, and `decays` is `dt * A`,
the log decay. An assignment is carried back into `dt`: `betas` becomes `dt =
log(expm1(betas)) - dt_bias`, and `decays` the `betas` it implies,
`decays / A`, so the kernel computes the gate you wrote and a read after the
write returns it. Assign them; an in-place edit (`mix.betas[:, t] = 0`) does not
reach `dt`. Three consequences:

- **Writing `decays` rewrites `betas` too.** `decays * 0.5` halves `dt`, so each token
  keeps more of the state *and* writes at half strength, and `decays = 0` ("keep
  everything") sets `dt` to zero, which writes nothing: the state stays at zero
  through a whole prompt. There is no way to change the decay alone.
- **`betas[:, t] = 0` is the exact "skip this token" on Mamba-2**: no write and
  `exp(0) = 1`, no decay, so the state after token `t` equals the state after `t - 1`
  bit for bit. Nemotron-H's scan clamps `dt` to at least `time_step_min` (0.001 on the
  released configs), so there a written 0 runs as 0.001 and the state still moves.
- **Writing the same values back is not a no-op.** The round trip through the
  softplus's inverse rounds `dt`: `mix.betas = mix.betas` on every block moves float32
  logits by about 2e-4 and bf16 logits by up to 1.8 on mamba2-130m (on block 0 alone
  the bf16 round trip happened to be exact). As a control, compare against an
  unedited run in float32, not against a write-back.

```python
with model.generate(prompt, max_new_tokens=2, do_sample=False) as tracer:
    for step in tracer.iter[1]:                        # a decode step
        entering = mix.state_input.save()
        mix.betas = torch.zeros_like(mix.betas)         # no write, and exp(0) = 1: no decay
        leaving = mix.state_output.save()               # == entering

with model.trace(prompt):
    betas = mix.betas.clone()
    betas[:, 3] = 0                                     # token 3 writes nothing and keeps everything
    mix.betas = betas
    logits = model.logits.save()

with model.trace(prompt):
    mix.decays = mix.decays * 0.5                       # half of dt: slower decay AND half-strength writes
    logits = model.logits.save()
```

`betas` must be positive or zero (zero stops the token's write and its decay; the inverse
of the softplus is undefined below it). On a prompt the chunk scan clamps
`dt` to `time_step_limit` after the softplus, so a written value outside the
limit runs clamped; a decode step does not clamp.

## The state after every token

The chunk scan computes the state at every chunk boundary in one tensor
(`new_states`, `[batch, chunks + 1, heads, head_dim, state_dim]`: the state
before each chunk and after the last). With a chunk size of 1 every token is
a boundary. `nnterp.chunk_per_token(model)` sets each Mamba-2 mixer's
`chunk_size` to 1, and then `states` and `state_after(t)` read the state after
every token of the call:

```python
from nnterp import chunk_per_token

chunk_per_token(model)                         # this model only; the logits change by rounding
with model.trace(prompt):
    states = mix.states.save()                 # [batch, seq, heads, state_dim, head_dim]
    final = mix.state_output.save()            # == states[:, -1]
with model.trace(prompt):
    s3 = mix.state_after(3).save()             # == states[:, 3]
chunk_per_token(model, False)                  # back to the chunk size the mixer was built with
```

Token by token, `states` is SSD's recurrence, from `state_input` (zeros on a
fresh prompt): `states[:, t] == exp(decays[:, t]) * states[:, t - 1] +
betas[:, t] * B_t ⊗ x_t`, with `B` repeated from groups to heads
(`tests/families/ssd.py`, `test_states_per_token`). On a decode step `states`
is that step's one token, `state_output` with a sequence axis of 1, and
`state_after(0)` is the same state.

`states` is a copy, `new_states[:, 1:]` transposed to the key-side-first
`States` layout, and read-only: an edit to it does not reach the scan.

Chunking per token computes the same scan in another order, so the logits move by
rounding: about 2e-4 in float32 on mamba2-130m, and up to 1.25 in bf16 on one prompt.
Compare runs made under the same chunking, or in float32.

- **Per model, not per family.** `chunk_per_token` sets an attribute of each
  mixer module of the model you pass, which the forward reads on every call;
  unlike `route_kernels` it touches no other model. `chunk_per_token(model,
  False)` restores the chunk size each mixer was built with
  (`config.chunk_size`, `mamba_chunk_size` on Bamba and Falcon-H1). Call it on
  a loaded model.
- **Slower on long prompts.** The recurrence between chunks is quadratic in
  the number of chunks, and with a chunk size of 1 that is the number of
  tokens.
- **Without it**, `states` reports `the chunk scan materializes the state only
  at chunk boundaries, every 256 tokens (chunk_size=256); call
  nnterp.chunk_per_token(model) to set every mixer's chunk_size to 1, so every
  token is a boundary (slower on long prompts)` (the mixer's own chunk size),
  and `state_after` raises `Unavailable` with it.

What stays unavailable, with or without it:

- `state`, the per-occurrence value a `tracer.iter` walk reads on a gated
  DeltaNet: `the chunk scan computes every token's state in one tensor per
  call, not one occurrence per token to walk with tracer.iter; read states, or
  state_after(t), after nnterp.chunk_per_token(model)`.
- `set_state_after`: `the chunk scan computes every boundary state in one
  cumulative step from the initial state, so a state written at token t does
  not flow into later tokens' states; assign state_input to change where a
  call starts`.

Both appear in `model.support()` under `linear_attn.state` and
`linear_attn.set_state_after`, and reading `mix.state` or `mix.set_state_after`
raises `Unavailable` with the reason.

## Two kernels, one value

A prompt runs `mamba2_chunk_scan`, and each decode step of `generate` runs
`mamba2_selective_state_update`, which takes its arguments in other places and
has no sequence axis. The forward decodes when `use_precomputed_states and
seq_len == 1`; the values read that binding and the call's length once per
call and select from the kernel that fires, and a decode step's tensors are
served with a sequence axis of 1. So the same value works in a `trace` and at
every step of `tracer.iter`, and the state hands off:

```python
entering, leaving = [], []                       # outside the block: a name bound inside does not survive it
with model.generate(prompt, max_new_tokens=3, do_sample=False) as tracer:
    for step in tracer.iter[:3]:
        s = mix.state_input
        entering.append(s.save() if s is not None else None)
        leaving.append(mix.state_output.save())

entering[0] is None                              # a fresh prompt
torch.equal(leaving[0], entering[1])             # step 1 starts from what step 0 left
```

On a decode step the update is SSD's recurrence over that step's values:
`state_output == exp(decays)[..., None, None] * state_input + betas[..., None,
None] * B ⊗ x`, with `B` repeated from groups to heads
(`tests/families/ssd.py`, `test_state_hands_off_under_generate`). The update
writes the new state into the cache and returns only `y`, so a decode step's
`state_output` is read inside it; assigning it changes both the cache and `y`.

## The kernels have to be the pure-torch ones

transformers dispatches both scans to `mamba_ssm`'s CUDA kernels when that
package is installed; they have no Python source, and every value but
`attention_output` reports `read inside transformers' pure-torch
mamba2_chunk_scan, but this process dispatches it to an optimized kernel
(mamba_ssm) with no Python source; uninstall it, or call
nnterp.route_kernels(model.family, 'torch'), to read these`. `route_kernels(family,
"torch")` binds the family's two kernel names to transformers' pure-torch
functions, process-wide; a prompt keeps the chunked scan. Call it before the
first trace that reads a value inside the mixer (a plain trace before it does
not fix the kernels); `route_kernels(family, "default")` restores the
optimized kernels. On a CPU the optimized kernels do not run at all, so a
model with `mamba_ssm` installed needs the routing before any trace on CPU:
unrouted, a `layer_output` read fails with `ValueError: Pointer argument cannot
be accessed from Triton (cpu tensor?)` when a GPU is visible and `RuntimeError:
invalid argument to exchangeDevice` when none is. Only the
two scan kernels are routed: the values are read at the scan call, so the
short convolution's kernel (`causal_conv1d`, when installed) does not affect
them.

## Gotchas

- **Route before the layer is traced**, with the family module before loading
  (`nnterp.families.mamba2`) or `model.family` after; see
  [recurrent-mixer-internals.md](../developing/recurrent-mixer-internals.md).
- **`betas` and `decays` are one argument.** Writing `decays` changes `betas`
  (`decays = 0` zeroes every write); assign, an in-place edit does not land; and a
  write-back of the unchanged values is not exact in bf16. A written `betas` must be
  positive or zero, and on a prompt it is clamped to `time_step_limit` by the kernel.
- **On a decode step read `state_output` before `attention_head_outputs`.** The
  decode kernel updates the state first and reads `y` from it; the other order is cut
  short with nnsight's `was never reached: the loop asked for a step the run did not
  make` warning, which blames the loop, not the order.
- **The state is not all a block remembers.** The short convolution before the scan
  (width 4) mixes each token with the three before it, so a state written through
  `state_input` leaves the last tokens in the convolution's window.
- **In-place edits on the queries, keys and values need `torch.no_grad()`**;
  assignment always works.
- **`chunk_per_token` is not undone by `route_kernels(family, "default")`**;
  call `chunk_per_token(model, False)`.
- **`state_input` is `None` on a fresh prompt.** Save it only when it is not.
- **A prompt run with `use_cache=False` returns no final state**: `state_output`
  raises `Unavailable` there.
- **Falcon-H1 with `mamba_rms_norm` off** passes the gate into the decode
  kernel, so a decode step's `attention_head_outputs` is gated by `silu(z)`
  where a prompt's is not. That is Falcon-H1-0.5B and the Falcon-H1-Tiny checkpoints
  (their `linear_attn.norm` is an identity); 1.5B and up set it.

## Related

- [delta-net.md](delta-net.md), the gated DeltaNet mixer with the same names, and a per-token state walked with `tracer.iter` and writable per token.
- [vocabulary.md](vocabulary.md), where `linear_attn` sits in the standard names.
- [availability.md](availability.md), per-block `support()` on a hybrid.
- [layouts.md](layouts.md), the layouts beside the softmax ones.
- [generation.md](generation.md), values per decode step.
- [recurrent-mixer-internals.md](../developing/recurrent-mixer-internals.md), how `StateSpace` sits on `RecurrentMixer`.
