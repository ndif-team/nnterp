---
title: Gated DeltaNet Hybrids
one_liner: The `linear_attn` values on Qwen3-Next, Qwen3.5, OLMo-Hybrid and Kimi-Linear blocks, the two kernels a prompt and a decode step run, and the per-token recurrent state behind `route_kernels`.
tags: [usage, hybrid, delta-net, linear-attention, state, qwen3_5_text, qwen3_next, kimi_linear]
related: [docs/usage/vocabulary.md, docs/usage/availability.md, docs/usage/layouts.md, docs/usage/attention-interior.md, docs/usage/generation.md, docs/usage/remote.md]
sources: [nnterp/components/linear_attention.py, nnterp/components/recurrent.py, nnterp/components/eproperty.py, nnterp/families/qwen3_5_text.py, nnterp/families/qwen3_next.py, nnterp/families/qwen3_5_moe_text.py, nnterp/families/olmo_hybrid.py, nnterp/families/kimi_linear.py, tests/families/test_qwen3_5_text.py, tests/families/test_kimi_linear.py]
---

# Gated DeltaNet Hybrids

## What this is for

Qwen3-Next, Qwen3.5 (text), Qwen3.5-MoE (text) and OLMo-Hybrid replace three blocks in four
with a gated DeltaNet mixer, `layers[i].linear_attn`; Kimi-Linear does the same with Kimi Delta
Attention, a gated DeltaNet whose decay is one per key channel
([below](#kimi-linear-kimi-delta-attention)). It projects queries,
keys and values like attention but mixes them through a per-head recurrent
state: each token decays the state by a learned gate, writes its key/value
pair in scaled by a beta, and the query reads against it. There is no pattern
and no scores. `nnterp.components.LinearAttention` gives such a block the same
names attention has where they mean the same thing, plus the gate, the beta
and the state entering and leaving the layer; and, routed through
transformers' token-by-token kernel, the state after every token of a prompt.
It is a `RecurrentMixer`, the base that reaches a recurrent mixer's values at
its kernel call, so the kernel switch and the per-token state below are the
base's.

> **How this mixer differs from Mamba-1 and Mamba-2.** The state is `[key_dim, value_dim]`
> per head, key side first. The update is the delta rule: each token decays the state by
> `exp(decays)` (one number per head), subtracts what the state already holds at its key
> and writes the remainder scaled by `betas`. `decays` and `betas` are separate kernel
> arguments, each writable on its own. The queries and keys are served *before* the
> kernel's l2-norm and `1/sqrt(key_dim)` scale. The per-token state can be read and
> written (`state`, `set_state_after`) once the family is routed through the torch
> kernels. A short convolution before the kernel (width 4) also carries the last few
> tokens. [vocabulary.md](vocabulary.md#same-name-different-meaning) puts the three
> mixers side by side.

## Canonical pattern

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("Qwen/Qwen3.5-9B", attn_implementation="eager")
prompt = "The Eiffel Tower is in the city of"

linear = [i for i, layer in enumerate(model.layers) if getattr(layer, "linear_attn", None) is not None]
softmax = [i for i, layer in enumerate(model.layers) if getattr(layer, "self_attn", None) is not None]
mix = model.layers[linear[0]].linear_attn

with model.trace(prompt):
    q = mix.attention_queries.save()        # [batch, seq, heads, key_dim]
    g = mix.decays.save()                   # [batch, seq, heads], float32, <= 0
    b = mix.betas.save()                    # [batch, seq, heads], in (0, 1)
    state = mix.state_output.save()         # [batch, heads, key_dim, value_dim]: after the last token
    out = mix.attention_output.save()       # [batch, seq, hidden]: what the mixer adds to the stream
    pattern = model.layers[softmax[0]].self_attn.attention_probabilities.save()   # the attention block still has one
```

Decide which blocks have which mixer *outside* the trace, as above.
`config.layer_types` says the same thing (`"linear_attention"` /
`"full_attention"`), and the tree follows it: a block has `self_attn` or
`linear_attn`, never both.

## Which blocks, and what `support()` says

`model.support()` reports each `self_attn` value as `no self_attn module on
this block` on the linear blocks and each `linear_attn` value as `no
linear_attn module on this block` on the attention blocks, so a hybrid's
`support()` reads as a short dict per value: `{3: 'no linear_attn module on this
block', 7: ..., ...}`. `model.support(layer=i)` is flat for one block.

## The values

| value | what it is | layout |
| --- | --- | --- |
| `attention_output` | what the mixer adds to the residual stream | `Residual`: `batch seq hidden` |
| `attention_queries`, `attention_keys` | what the delta rule receives: after the short convolution, the activation and the repeat to the value heads, before the kernel's l2-norm (and, for the queries, its `1/sqrt(key_dim)` scale) | `LinearQK`: `batch seq heads key_dim` |
| `attention_values` | the values the delta rule receives | `LinearV`: `batch seq heads value_dim` |
| `decays` | the gate: the log of how much of the state each token keeps; float32, non-positive | `Gates`: `batch seq heads`; on Kimi-Linear `ChannelGates`: `batch seq heads key_dim`, one per key channel |
| `betas` | how strongly each token's key/value pair is written into the state; in `(0, 1)`, or `(0, 2)` on OLMo-Hybrid with `linear_allow_neg_eigval` (the released checkpoints) | `Gates`: `batch seq heads` |
| `state_input` | the state the call starts from: `None` on a fresh prompt, the cached state on a decode step (a copy) | `State`: `batch heads key_dim value_dim` |
| `state_output` | the state after the call's last token: what the next decode step starts from | `State`: `batch heads key_dim value_dim` |
| `attention_head_outputs` | each head's read of the state, before the gated norm and the output projection | `LinearV`: `batch seq heads value_dim` |

The layout names are the aliases in `nnterp.components` (`LinearAttention.decays.layout
is Gates`); `state` is a `State` and `states` a `States`, `batch seq heads key_dim
value_dim`. `heads` is the mixer's `num_v_heads` (the queries and keys are repeated up to
it), `key_dim` its `head_k_dim` and `value_dim` its `head_v_dim` (on Kimi-Linear, `heads` is
`num_heads` and `key_dim` and `value_dim` are both `head_dim`). The state is
float32 in the torch kernels. Everything but `attention_output` is read at the
delta-rule kernel call, so these are `EProperty` values keyed inside the
forward (`kernel("inputs")`, the kernel that fires on this call); assign to
replace them, or edit in place (`mix.attention_head_outputs[:, -1] = 0`
reaches the model). The queries, keys and values are views torch refuses to edit
in place while autograd is on (`RuntimeError: Output 0 of Select is a view and is
being modified inplace`): assign them, or edit in place under `torch.no_grad()`;
`attention_head_outputs`, `decays` and `betas` take in-place edits either way.
`state_input` is a clone of the cache's buffer: the cache
hands the kernel its own tensor and overwrites it with the new state
afterwards, so the live one would read as the step's *output* by the time the
trace ends.

### Recomputing the recurrence by hand

The kernel l2-normalizes the queries and keys and scales the queries by
`1/sqrt(key_dim)` before the loop (`use_qk_l2norm_in_kernel`, on Qwen3-Next and
Qwen3.5), so a recurrence written on the served tensors is off until it does the same
(a 42% relative error in the final state on Qwen3.5-0.8B). With it, it matches `states`
and `attention_head_outputs`:

```python
l2norm = lambda t: t * torch.rsqrt((t * t).sum(-1, keepdim=True) + 1e-6)

with model.trace(prompt):
    q, k, v = mix.attention_queries.save(), mix.attention_keys.save(), mix.attention_values.save()
    g, b = mix.decays.save(), mix.betas.save()
    y = mix.attention_head_outputs.save()

q, k = l2norm(q.float()) / q.shape[-1] ** 0.5, l2norm(k.float())    # what the kernel uses
S = torch.zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3])      # [batch, heads, key_dim, value_dim]
outs = []
for t in range(q.shape[1]):
    S = S * g[:, t].exp()[..., None, None]                            # decay
    write = (v[:, t] - (S * k[:, t, ..., None]).sum(-2)) * b[:, t, :, None]   # what the state lacks at this key
    S = S + k[:, t, ..., None] * write[..., None, :]
    outs.append((S * q[:, t, ..., None]).sum(-2))                     # the query's read
torch.testing.assert_close(torch.stack(outs, 1), y.float(), rtol=1e-4, atol=1e-4)
```

## Kimi-Linear: Kimi Delta Attention

Kimi-Linear's mixer (`KimiLinearDeltaAttention`) runs the gated DeltaNet's forward with
transformers' own kernels, `chunk_kimi_delta_attention` on a prompt and
`recurrent_kimi_delta_attention` on a decode step, so every value above means the same
thing and `route_kernels(model.family, "torch")` gives `state` and `states` as on
Qwen3.5. What differs is the decay: the forget gate gives one log decay per key channel,
so `decays` is `[batch, seq, heads, key_dim]` (`ChannelGates`) and each row of the state
decays on its own. The recurrence above changes in that one line, `S = S *
g[:, t].exp()[..., None]`, and matches `attention_head_outputs` with it (on the pinned
tiny checkpoint's copy):

```python
l2norm = lambda t: t * torch.rsqrt((t * t).sum(-1, keepdim=True) + 1e-6)

with model.trace(prompt):
    q, k, v = mix.attention_queries.save(), mix.attention_keys.save(), mix.attention_values.save()
    g, b = mix.decays.save(), mix.betas.save()                        # g: [batch, seq, heads, key_dim]
    y = mix.attention_head_outputs.save()

q, k, v = l2norm(q.float()) / q.shape[-1] ** 0.5, l2norm(k.float()), v.float()
S = torch.zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3], device=q.device)
outs = []
for t in range(q.shape[1]):
    S = S * g[:, t].exp()[..., None]                                   # decay, one factor per key channel
    write = (v[:, t] - (S * k[:, t, ..., None]).sum(-2)) * b[:, t, :, None]
    S = S + k[:, t, ..., None] * write[..., None, :]
    outs.append((S * q[:, t, ..., None]).sum(-2))
torch.testing.assert_close(torch.stack(outs, 1), y.float(), rtol=1e-2, atol=1e-5)
```

transformers names both of Kimi-Linear's mixers `self_attn`. The family aliases the KDA
module `linear_attn` and sets `self_attn` to `None` on its blocks, so, as on the other
hybrids, a block has `self_attn` (the latent attention) or `linear_attn`, and
`getattr(layer, "self_attn", None) is not None` picks the attention blocks. The KDA
module's own path is still `model.model.layers[i].self_attn` in error messages and
`.path`; reach it as `layers[i].linear_attn`.

## Two kernels, one value

A prompt runs `torch_chunk_gated_delta_rule`, and each decode step of
`generate` runs `torch_recurrent_gated_delta_rule`: two different operations
in the forward, chosen by the forward's test `use_precomputed_states and
seq_len == 1`. The values read that binding and the call's length and name
the call that fires on this step (`RecurrentMixer.KERNEL`: the recurrent
kernel for one token over a cached state, the chunked one otherwise), so
the same value works in a `trace` and at every step of
`tracer.iter`, and the state hands off from one step to the next:

```python
entering, leaving = [], []                       # outside the block: a name bound inside does not survive it
with model.generate(prompt, max_new_tokens=3, do_sample=False) as tracer:
    for step in tracer.iter[:3]:
        s = mix.state_input
        entering.append(s.save() if s is not None else None)
        leaving.append(mix.state_output.save())

entering[0] is None                              # a fresh prompt
torch.equal(leaving[0], entering[1])             # step 1 starts from what step 0 left
# inside the loop, mix.attention_queries.shape[1] is prompt_len on step 0, then 1 per step
```

The kernels have to be transformers' pure-torch ones. With
`flash-linear-attention` installed, the delta-rule kernels dispatch to a
compiled kernel with no Python source, and every kernel value reports `read
inside transformers' pure-torch torch_chunk_gated_delta_rule, but this process
dispatches it to an optimized kernel (fla) with no Python source; uninstall it,
or call nnterp.route_kernels(model.family, 'torch'), to read these`.
`causal-conv1d` alone replaces only the short convolution before the kernel;
the kernel values stay readable, since only the delta-rule kernels are checked.

## The state after every token

The chunked kernel carries the state between 64-token chunks and never
materializes it per token. transformers' token-by-token kernel does, at a
cost, and like eager attention that is a choice made before tracing:

```python
import nnterp
from nnterp import StandardizedTransformer, route_kernels

route_kernels(nnterp.families.qwen3_5_text, "torch")     # process-wide, like installing a kernel
model = StandardizedTransformer("Qwen/Qwen3.5-9B", attn_implementation="eager")
mix = model.layers[0].linear_attn
```

`route_kernels(family, kernel)` binds both kernel names in the family's
modeling module to the token-by-token torch loop (`"torch"`) or back to what
the module bound at import (`"default"`); `route_delta_rule(family,
"recurrent" | "chunked")` is the same switch in the delta rule's words.
`family` is the family module (`nnterp.families.qwen3_5_text`, or
`model.family` on a loaded model) or the modeling module. Call it before the
layer's forward is traced: loading, then `route_kernels(model.family,
"torch")`, then tracing works; a model
whose layer has already been traced keeps the kernel it was instrumented with
(see Gotchas). It applies to every model of the family in the process. The two
kernels compute the same rule: the logits agree to float error.

Then four things exist. `state` is the state after one token of the prompt,
one occurrence per token, so nnsight's own iteration walks it:

```python
per_token = []
with model.trace(prompt) as tracer:
    for t in tracer.iter[:]:                   # every token of the prompt
        per_token.append(mix.state.save())     # [batch, heads, key_dim, value_dim] each

with model.trace(prompt) as tracer:
    for t in tracer.iter[7]:
        mix.state = torch.zeros_like(state)    # a write at token 7: tokens 8.. continue from zeros
    for t in tracer.iter[8]:
        s8 = mix.state.save()                  # the state after token 8 (the last of 9), downstream of the write
```

A position must lie inside the prompt: `tracer.iter[9]` on this 9-token prompt is never
reached, and the trace is cut short there with a `was never reached` warning, leaving
`s9` unbound.

Outside any `tracer.iter`, `mix.state` is the state after token 0. A read and a
write of the same token in one body work: `mix.state = mix.state * 0` inside
`tracer.iter[7]` zeroes the state after token 7, the same as
`mix.set_state_after(7, torch.zeros_like(...))`. `states` is every position stacked,
read-only:

```python
with model.trace(prompt):
    states = mix.states.save()                 # [batch, seq, heads, key_dim, value_dim]
    final = mix.state_output.save()            # == states[:, -1]
```

`state_after(t)` and `set_state_after(t, value)` are the two `tracer.iter`
forms above as calls, and they compose in one trace when the reads respect
the write:

```python
with model.trace(prompt):
    before = mix.state_after(3).save()                       # a position before the write: read it first
    mix.set_state_after(7, mix.state_after(7) * 0)           # read-and-write at the same token, as calls
    after = mix.state_after(8).save()                        # a position after the write: read it after
    logits = model.logits.save()
```

Two things a state write does that "change the memory from token `t` on" does not say:

- **It changes token `t`'s own output.** The state after token `t` is what token `t`'s
  query reads, so `set_state_after(t, value)` moves `attention_head_outputs[:, t]` too
  (and nothing before it).
- **It is not all the block remembers.** The short convolution in front of the kernel
  (width 4) mixes each token's queries, keys and values with the three tokens before
  it, so `set_state_after(t, zeros)` is not a clean "forget everything before `t`":
  tokens `t + 1` to `t + 3` still see tokens up to `t` through the convolution.

Under `generate` the prompt is one kernel call and each decode step another,
over one token, so the per-token view is an inner loop on step 0 and then one
state per step:

```python
n = len(model.tokenizer(prompt).input_ids)
per_token = []
with model.generate(prompt, max_new_tokens=3, do_sample=False) as tracer:
    for step in tracer.iter[:3]:
        if step == 0:
            for t in tracer.iter[:n]:              # the prompt's tokens
                per_token.append(mix.state.save())
        else:
            per_token.append(mix.state_output.save())   # the one token this step processes
# len(per_token) == n + 2; per_token[:n] equals a trace's `states` token for token
```

`states`, `state_after` and `set_state_after` count from the current call's
own first token on every step: on a decode step `states` is `[batch, 1, ...]`
and `state_after(0)` is that step's token.

### Read order

Reads follow the forward. In one trace, positions before a write are read
before it and positions after it are read after; `states` reads every
position, so it goes in a trace of its own. A read that
asks for a position the run has already passed does not raise: nnsight cuts
the block short there with a `UserWarning` (`'...last_recurrent_state_3.output.i7'
was never reached: the loop asked for a step the run did not make, so it was
cut short`), keeps what was saved before it, and skips every statement after
it, so a name bound later is undefined when the block exits.

### `support()` reasons

Without the switch, `state` and `states` report `the state after each token
is materialized only by the token-by-token kernel; the chunked kernel a prompt
runs through carries it between chunks. Call
nnterp.route_kernels(model.family, 'torch') before tracing this layer
(slower, like attn_implementation='eager')`, and reading one, or calling
`state_after` / `set_state_after`, raises `Unavailable` with it. With an
optimized kernel installed they report the kernel reason above, like every
other value. On the attention blocks they report `no linear_attn module on
this block`.

## Gotchas

- **In-place edits on the queries, keys and values need `torch.no_grad()`**; assignment
  always works.
- **`state` takes assignment, not in-place edits.** `mix.state[:] = 0` raises
  `OutOfOrderError`; assign a tensor.
- **Route before the layer is traced.** `route_kernels` after a model has
  already traced that layer leaves the forward on the kernel it was
  instrumented with: `states` then raises `AttributeError: 'LinearAttention'
  object (nor its module) has attribute 'states'` and `state` raises
  `SourceNotAvailable` naming `last_recurrent_state_3` as missing under the
  chunked kernel. Route, then load or trace.
- **`model.family` needs a loaded model.** Route with the family module
  (`nnterp.families.qwen3_5_text`) before loading, or with `model.family` after
  loading and before the first trace.
- **`states` is read-only** (`AttributeError: states is derived and read-only`);
  `set_state_after` writes one position.
- **`getattr(layer, "self_attn", None)` inside a trace can trip served values.**
  Decide the block lists outside the trace.
- **`state_input` is `None` on a fresh prompt.** Save it only when it is not.
- **`tracer.result` after an open `tracer.iter[:]` is never bound**; use a
  bounded loop ([generation.md](generation.md)).

## Related

- [vocabulary.md](vocabulary.md), where `linear_attn` sits in the standard names.
- [availability.md](availability.md), per-block `support()` on a hybrid.
- [layouts.md](layouts.md), the `linear_attn` layouts beside the softmax ones.
- [attention-interior.md](attention-interior.md), the same names on the softmax blocks, with a pattern and scores.
- [generation.md](generation.md), values per decode step, and the loop rules under `generate`.
- [remote.md](remote.md), what these source-located values need from a server.
- nnsight docs/usage/iter-all-next.md and docs/gotchas/iteration.md, `tracer.iter` and the cut-short loop.
