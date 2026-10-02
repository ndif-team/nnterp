---
title: Recurrent Mixer Internals
one_liner: How RecurrentMixer reaches a recurrent mixer's values — the kernel op the forward's own test picks and its once-per-call record (`KERNEL`, `per_call`), the pure-torch kernels .source needs, process-wide kernel routing, and per-token state through occurrence arithmetic — and how LinearAttention (gated DeltaNet), SelectiveScan (Mamba-1) and StateSpace (Mamba-2) sit on it.
tags: [developing, internals, hybrids, deltanet, linear-attention, mamba, selective-scan, state-space, recurrent, occurrences]
related: [docs/developing/eproperty-internals.md, docs/developing/architecture.md, docs/developing/gotchas.md, docs/usage/delta-net.md, docs/usage/selective-scan.md, docs/usage/state-space.md]
sources: [nnterp/components/recurrent.py, nnterp/components/linear_attention.py, nnterp/components/selective_scan.py, nnterp/components/state_space.py, nnterp/components/layer.py, nnterp/families/mamba.py, nnterp/families/nemotron_h.py, tests/families/scan_suite.py, tests/families/ssd.py, nnterp/components/eproperty.py, nnterp/families/qwen3_5_text.py, tests/families/test_qwen3_5_text.py, tests/test_base.py, nnsight src/nnsight/intervention/interleaver.py, nnsight src/nnsight/intervention/iterator.py]
---

# Recurrent Mixer Internals

## What this is for

A recurrent mixer (a gated DeltaNet, a state-space layer) runs its sequence
through a kernel function its forward calls, and its values are the
arguments and results of that call. The forward has three properties the
softmax `Attention` never meets: it calls a **different** kernel on a prompt
and on a decode step, the kernel is a **module global** that transformers
rebinds to an optimized implementation when one is installed, and the state
after every token exists, if at all, only inside a **loop** in one of the
two kernels. How a mixer's values are reached through all three is the same
whatever the values are, so it lives in one base class and the mixers
declare only their values:

- `RecurrentMixer` (`nnterp/components/recurrent.py:257-446`) holds the
  mechanism: the kernel choice, `attention_output`, the availability
  predicates, the per-token state machinery and the routing functions.
- `LinearAttention` (`nnterp/components/linear_attention.py:23-96`), the
  gated DeltaNet mixer (`linear_attn` on Qwen3-Next, Qwen3.5, Qwen3.5-MoE
  text and OLMo-Hybrid), sets the kernel constants and declares its eight
  values: `attention_queries`, `attention_keys`, `attention_values`,
  `decays`, `betas`, `state_input`, `attention_head_outputs`,
  `state_output`.
- `SelectiveScan` (`nnterp/components/selective_scan.py`), the Mamba-1 mixer
  (`linear_attn` on Mamba, Falcon-Mamba and Jamba's Mamba blocks), declares
  the same names over a selective scan; the section
  [Mamba-1: `SelectiveScan`](#mamba-1-selectivescan) is what it adds.

This page is how the base handles each property, told through the DeltaNet
subclass with the transformers lines it depends on
(`models/qwen3_5/modeling_qwen3_5.py` on the stack
[transformers-compat.md](transformers-compat.md) names; Qwen3-Next is the
same shape).

## Canonical pattern

Occurrences are counted per location since the op was drilled into in the
run, so a decode step's first token is not occurrence 0 once an earlier step
has fired the same op.
Run on `yujiepan/qwen3.5-tiny-random`:

```python
import torch
import nnterp
from nnterp import StandardizedTransformer, route_kernels
from nnterp.families import qwen3_5_text

route_kernels(qwen3_5_text, "torch")                  # before any trace of a DeltaNet layer
model = StandardizedTransformer("Qwen/Qwen3.5-9B", dispatch=True, attn_implementation="eager")
mix = model.layers[0].linear_attn
prompt = "Hello world there"
n = len(model.tokenizer(prompt).input_ids)

with model.trace(prompt):
    prompt_states = mix.states.save()                 # [batch, n, heads, key_dim, value_dim]

firsts, seqs, stacks, outs = [], [], [], []           # bound outside: names bound in the block do not survive it
with model.generate(prompt, max_new_tokens=3, do_sample=False) as tracer:
    for step in tracer.iter[:]:
        op, first, seq = mix._token_op()              # (the state op, occurrence of this call's first token, its length)
        seqs.append(seq)
        firsts.append(first)
        stacks.append(mix.states.save())              # through whichever kernel fires this step
        outs.append(mix.state_output.save())

seqs    == [n, 1, 1]                                  # the prompt, then one token per step
firsts  == [0, 0, 1]                                  # the prompt's op starts at 0; step 1 at 0 of the *recurrent* op; step 2 at 1
torch.equal(stacks[0], prompt_states)                 # True
torch.equal(stacks[2][:, 0], outs[2])                 # True: step 2's `states` is step 2's token ...
torch.equal(stacks[2][:, 0], outs[1])                 # False: ... not occurrence 0 of the op, which was step 1's
route_kernels(qwen3_5_text, "default")
```

The open `tracer.iter[:]` ends with nnsight's "never reached" warning by
design (`nnsight iterator.py:66-71`). Without the `first` offset, step 2
would ask for occurrence 0 of the recurrent kernel's state op, which step 1
already consumed, and the read would dangle.

## Two kernels, one test

The mixer's forward binds `use_precomputed_states` before it branches
(`modeling_qwen3_5.py:561-563`), then calls
`torch_recurrent_gated_delta_rule(...)` when it has a cached state and one
token (`:625-637`) and `torch_chunk_gated_delta_rule(...)` otherwise
(`:638-650`). Both take `(query, key, value, g=, beta=, initial_state=, ...)`
and return `(core_attn_out, last_recurrent_state)`. nnsight names them
`torch_recurrent_gated_delta_rule_0` and `torch_chunk_gated_delta_rule_0`,
and the binding `use_precomputed_states_0` (a binding is an op,
nnsight `source.py:26-31`). The call's length is the sequence axis of the
masked input the forward binds once per call,
`apply_mask_to_padding_states_0`.

A subclass names these in class constants (`recurrent.py:296-309`):

| constant | meaning | `LinearAttention` |
|---|---|---|
| `BRANCH` | the binding the forward makes before it branches, `True` when the call continues from a cached state | `"use_precomputed_states_0"` (the base's default) |
| `SEQ_OP` | the forward's masking of its input, once per call: `[batch, seq, ...]`, the call's length | `"apply_mask_to_padding_states_0"` (the base's default) |
| `CHUNK_KERNEL` | the call a prompt runs through | `"torch_chunk_gated_delta_rule_0"` |
| `RECURRENT_KERNEL` | the call a decode step runs through | `"torch_recurrent_gated_delta_rule_0"` |
| `STATE_OP` | inside the token-by-token kernel, the binding of the state after each token's update, or `None` | `"last_recurrent_state_3"` |
| `STEP_STATE_OP` | set when the decode kernel is a single-step update, not the token loop: the binding of the new state inside it; the prompt's kernel is then the loop | `None` (Mamba-1: `"ssm_state_0"`) |

- `KERNEL` (`:311-329`) is one `staticmethod` on the base, the same for
  every mixer: the forward's own test, `use_precomputed_states and seq_len
  == 1`. It reads `BRANCH` and `SEQ_OP` in the order the forward makes them
  (sorted by each op's source line: the DeltaNet forward masks its input
  before it binds the branch variable, the Mamba forwards after) and
  returns `RECURRENT_KERNEL` when the call continues a cached state and has
  one token, `CHUNK_KERNEL` otherwise: a prompt, or several tokens over a
  cached state
  (`tests/test_base.py::test_cached_call_with_several_tokens_reads_the_prompts_kernel`).
  Being a `staticmethod`, `type(self).KERNEL(self)` and
  `self.KERNEL(self)` are the same call.
- `kernel(attribute)` (`:247-254`) is the key function every
  kernel-located value is declared on: it returns
  `source.<KERNEL(envoy)>.<attribute>`, so `LinearAttention` declares its
  values as `@EProperty(kernel("inputs"), select=0, ...)` through
  `@EProperty(kernel("output"), select=1, ...)`
  (`linear_attention.py:51-95`). At read time `KERNEL` reads the two
  bindings' `.output` on this call and picks the name. The choice is made
  once per call ([Once per call](#once-per-call-kernel-and-per_call)), so
  the eight values read in one step ask the model for the bindings once.
- `attention_output` (`:341-348`) is the base's: the module's own output,
  the contribution to the residual stream, unwrapped from a tuple and
  rewrapped on write.

Each decode step under `generate` is one forward over one token: the
kernel's inputs then have sequence length 1, `state_input` is the cached
state, and `state_output` the state the step leaves
(`tests/families/test_qwen3_5_text.py:93-109`).

## Once per call: `KERNEL` and `per_call`

The bindings the forward tests are served once per module call, and several
values of one call depend on the kernel they pick, so the decision is made
once and kept for the call:

- `RecurrentMixer.KERNEL` (`recurrent.py:311-329`) makes its two reads and
  its choice inside `per_call`, under the key `"kernel"`.
- `per_call(envoy, key, compute)` (`:190-228`), exported from
  `nnterp.components`, keeps one record per `(envoy.path, key)`,
  `(call, value)`, and runs `compute()` when there is no record or when the
  record's call is not the current one (`:226-227`). The records live on
  the worker, the greenlet running the intervention code
  (`getcurrent().__dict__["_nnterp_per_call"]`). Every run of every invoke
  has its own worker, a replayed `model.edit` too, so two invokes reading
  one mixer do not share a record
  (`tests/test_base.py::test_two_invokes_read_one_mixers_values`), an
  edit replayed on another prompt reads that call's values
  (`::test_edit_on_a_kernel_value_replays_on_another_prompt`), and nothing
  stays on the envoy: the tensors in a record are freed with the worker.

One rule names the current call (`:216`):

```python
call = mediator.iteration or mediator.occurrence(f"{envoy.path}.output")
```

- **Pinned to step k ≥ 1.** A read pinned by `tracer.iter` to step k is
  served at the k-th occurrence of its location, and a mixer's kernel fires
  once per call, so the read is in call k.
- **Relaxed, on step 0, or outside `tracer.iter`.** `iteration` is `None`
  after a step's first read and `0` on step 0 and in a plain trace, so the
  call is how many times the module's `.output` has been passed. nnsight
  counts every location's visits per worker whether or not the location was
  read (`interleaver.py:331-334`, `occurrence`): inside call c, c calls
  have returned, and between calls the count names the one about to start.

The two agree on step k of a `generate`, so every read of a step body finds
the same record whichever comes first: a body that reads `state_input`
first (pinned) and `states` second (relaxed) computes this step's `(op,
first, seq)` under `"call"`, and a body whose first read is not a value of
the mixer (`linear_attn.input`, a plain nnsight read that relaxes the pin)
still decides its own kernel, from the count
(`tests/test_base.py::test_decode_step_whose_first_read_is_relaxed_takes_its_own_branch`).
`compute()` may park the worker until the model reaches what it reads; the
record is filed under the call decided before it runs, the one that read
lands in.

`per_call` holds three things: `KERNEL`'s choice (key `"kernel"`),
`_token_op`'s `(op, first, seq)` (key `"call"`) and the kernel call's
`(args, kwargs)` that `StateSpace._arguments` reads (key `"arguments"`,
`state_space.py:213-221`).

## `state_input` is a clone

The forward passes the cache's own buffer as `initial_state`
(`modeling_qwen3_5.py:624`, `cache_params.layers[i].recurrent_states[0]`)
and afterwards `cache_params.update_recurrent_state(last_recurrent_state,
...)` (`:653-654`) writes the new state into that buffer in place
(`transformers/cache_utils.py:1091-1093`, "Update the linear attention
cache in-place"). A saved `state_input` that held the live tensor would read
as this step's *output* by the time the trace ends, so the preprocess
returns `value.clone()` (`linear_attention.py:76-85`). Assigning replaces
what the step starts from; the clone only affects reads. It is a DeltaNet
value, so it is the subclass's; a mixer whose kernel takes the cached state
the same way clones it the same way.

## Optimized kernels have no source

transformers decorates both kernels with
`use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "fla")` /
`("fused_recurrent_gated_delta_rule", "fla")` (`modeling_qwen3_5.py:300`,
`:437`); Mamba-2's `mamba2_chunk_scan` / `mamba2_selective_state_update`
carry the same decorator over `mamba_ssm`. The decorator
(`transformers/integrations/hub_kernels.py:829-870`) binds the
module-level name to a `wrapped` closure whose nonlocals are
`torch_function` (the pure-torch body) and `implementation` (the optimized
function when its package is installed, else `torch_function` again).

- `_dispatch(bound)` (`recurrent.py:46-53`) reads that closure with
  `inspect.getclosurevars(bound).nonlocals`, `{}` for a plain function;
  `_torch_function(bound)` (`:56-58`) is its `torch_function`, or the
  binding itself. `_name(op)` (`:41-43`) is the module-level name an op
  calls (`torch_chunk_gated_delta_rule_0` -> `torch_chunk_gated_delta_rule`).
- `needs_torch_kernels` (`:146-160`) is the `unavailable` predicate of every
  kernel-located value: for `type(envoy).CHUNK_KERNEL` and
  `RECURRENT_KERNEL` it refuses when `implementation is not
  torch_function`. A compiled kernel has no Python source, so `.source`
  could not drill into it (nnsight `source.py:364-374`, `compiled` raises
  `SourceNotAvailable` for a callable without `__code__`). The reason
  names the package and says to uninstall it or route to the torch kernels.

## Routing the kernels: `route_kernels`

`route_kernels(family, kernel)` (`:95-129`) rebinds the family's kernel
names in its modeling module, process-wide:

- `_mixer(family)` (`:65-92`) finds the transformers modeling module and
  the mixer's envoy class from the family's `ENVOYS` entry whose envoy
  subclasses `RecurrentMixer`. Given a modeling module directly, it takes
  the `RecurrentMixer` subclass a loaded family keys on one of that
  module's classes, else one whose `CHUNK_KERNEL` the module defines.
- The module's original bindings of both names are stashed once in
  `module.__dict__["_nnterp_kernels"]`, so `"default"` restores them and
  repeated `"torch"` calls are idempotent.
- `"torch"` binds each name to its own pure-torch kernel, and on a mixer
  with a `STATE_OP` the prompt's name to the token loop, `_loop_kernel()`'s
  `torch_function`: `RECURRENT_KERNEL`'s on a gated DeltaNet, so **both**
  names are bound to the token-by-token kernel; `CHUNK_KERNEL`'s own on
  Mamba-1 (`STEP_STATE_OP` set), whose pure-torch scan is the loop. On the
  DeltaNet: the forward's op names stay
  `torch_chunk_gated_delta_rule_0` / `torch_recurrent_gated_delta_rule_0`
  whatever the globals hold (the name in the source is the label, the
  global is what runs), so after routing a prompt's
  `torch_chunk_gated_delta_rule_0` *is* the token loop and `STATE_OP` under
  it has one occurrence per token. On a mixer without a `STATE_OP`, each
  name is bound to its own `torch_function`: the kernels become readable
  and a prompt keeps its chunked kernel.
- `route_delta_rule(family, kernel)` (`:132-143`) is the DeltaNet spelling:
  `"recurrent"` is `"torch"`, `"chunked"` is `"default"`.

**Why it must run before tracing that layer.** nnsight's `.source` builds
the instrumented forward once per module and, when it drills into a call,
instruments the callee it received and stores it in `interleaver.sourced`
for the run; `instrument` rebuilds the function with `fn.__globals__`
copied into a new function object (nnsight `source.py:439-444`,
`function_like`). A callee already instrumented keeps the binding it was
compiled with, and the module-level `forward` body has already been
compiled over its globals dict, so a rebinding made after the first drill is
not what the instrumented copy calls. The reason and the docstring
say so; the tests route, load, trace, and restore in a `finally`
(`test_qwen3_5_text.py:118-148`), and `tests/test_base.py` checks that
`"torch"` / `"default"` round-trip the bindings.

## The state after every token

Only the recurrent kernel has a per-token state. Its loop
(`modeling_qwen3_5.py:480-491`) rebinds `last_recurrent_state` twice per
token, decay at `:484` and update at `:489`; with the two bindings before
the loop (`:474`, `:476`) the post-update binding is the **fourth**
`last_recurrent_state` in the function, so `STATE_OP =
"last_recurrent_state_3"` (`linear_attention.py:49`) fires once per token.
The chunk kernel binds the same name four times too (`:407`, `:409`, `:425`
inside the chunk loop, `:428`), so `last_recurrent_state_3` exists there as
well, but it is the single final binding, not a per-token one; that is why
`state` and `states` are guarded by `needs_recurrent_routing`
(`recurrent.py:163-187`) rather than by the op resolving. The predicate
answers, in order:

1. `STATE_OP is None`: "this mixer's kernels do not materialize the state
   per token". A subclass without a per-token kernel declares nothing; its
   `state`, `states`, `state_after` and `set_state_after` report that.
2. `needs_torch_kernels`' reason, when an optimized kernel is bound.
3. The live binding of `CHUNK_KERNEL`'s name is not the token-by-token
   loop (or, with `STEP_STATE_OP`, `RECURRENT_KERNEL`'s is still the
   dispatcher, whose body has no state binding): the instruction to call
   `route_kernels(model.family, 'torch')`. Checked on the live binding, so
   `support()` follows the routing.

## Occurrence arithmetic

`state` is an `EProperty` whose key function returns
`source.{kernel}.source.{STATE_OP}.output` (`:352-377`), a location with
one occurrence per token, so nnsight's own `tracer.iter` walks it:
`for t in tracer.iter[:n]: mix.state` reads the state after every prompt
token, `tracer.iter[4]` the one after token 4, and an assignment there is a
write the following tokens continue from
(`test_qwen3_5_text.py:150-178`). `states`, `state_after` and
`set_state_after` are the same location addressed by index, and need three
numbers.

- **Which kernel.** `_token_state_op` (`:352-362`) is `state`'s key
  function. Under `tracer.iter` the pin at the read is a token index, while
  the bindings `KERNEL` reads and the kernel call's `.fn` fire once per
  forward. So inside `with pinned(None):` it decides the call's kernel
  like every other value (once per call, cached) and drills into the
  kernel call, relaxed; then it returns the key, and the pinned read that
  follows asks for that token's occurrence of the state op. This is what
  lets `for t in tracer.iter[2]: mix.state` resolve on a first read pinned
  past 0 (`test_qwen3_5_text.py:173-176`).
- **Where this call starts.** `_token_op` (`:383-403`) computes, once per
  call through `per_call`, `(op, first, seq)`: `op` is the state op of the
  kernel that fires (`_state_op(kernel)`, under the drilled kernel call),
  `seq` is `_seq()`, and `first` is
  `Mediator.current(location).occurrence(location)` for the state op's
  `.output` location. `_seq()` (`:379-381`) is an overridable method; the
  base reads `attention_queries.shape[1]`, the DeltaNet kernel's sequence
  axis, and a mixer whose kernel is laid out otherwise overrides it. The
  trick is *when* it is read. Reading a kernel argument parks the worker at
  the kernel call's start (nnsight `interleaver.py:331-334`, `occurrence`
  is the interleaver's count minus the count when this worker started),
  which is the one moment the state op's count is exactly the number of
  tokens **earlier calls** put through it. Occurrences are counted per
  location since the op was drilled into in the run (`interleaver.py:629-633`,
  `:785-791`), so on step *k* ≥ 1
  of a `generate` the recurrent op's count is *k* − 1: the canonical
  pattern's `[0, 0, 1]`. The prompt's tokens are under the chunk kernel's
  op, a different location, so they do not offset a decode step.
- **Reading position `t`.** `_states` (`:405-413`) loops `for t in
  range(seq): with pinned(first + t): states.append(op.output)` and stacks
  on axis 1; `state_after(t)` (`:429-434`) reads one; `set_state_after`
  (`:436-446`) writes one. `pinned(n)` (`:231-244`, exported from
  `nnterp.components`) is a context manager: it sets the worker mediator's
  `iteration` to `n`, as `tracer.iter[n]` does (`None` relaxes the pin),
  and restores the pin the worker had on the way out. The first hit relaxes
  the mediator (`interleaver.py:478-483`), so a `with` holds one read, and
  `_token_op` must have parked at the call's start before the loop begins;
  a second per-token read in the same call reuses its record and does not
  re-ask.
- `states` is a `DerivedEProperty` (`:418-422`): read-only, a stack of the
  per-occurrence reads. `_require_state` (`:424-427`) gives `state_after`
  and `set_state_after` the same `Unavailable` a read of `state` would
  raise.

Reads follow the forward: in one trace, positions before a write come
before it and positions after it come after; `states` reads every position,
so it goes in its own trace
(`test_qwen3_5_text.py:133-144`).

## Mamba-1: `SelectiveScan`

transformers' Mamba mixer (`models/mamba/modeling_mamba.py`; Falcon-Mamba's
and Jamba's are copies in their own modules) computes `dt`, `B`, `C` from the
conv output and calls `mamba_selective_scan(x, dt, A, B, C, D=, z=,
delta_bias=, ...)` on a prompt, `mamba_selective_state_update(state, x, dt,
A, B, C, D, z=, dt_bias=, ...)` when `use_precomputed_states and seq_len ==
1`. Both are decorated with the same hub dispatcher over `mamba_ssm`. Four
things differ from the DeltaNet, and each is handled where it arises:

- **The bindings come in the other order.** The forward binds
  `use_precomputed_states_0` before it masks its input
  (`apply_mask_to_padding_states_0`), the reverse of the DeltaNet's. The
  base's `KERNEL` reads the two sorted by source line, so `SelectiveScan`
  declares no `KERNEL` of its own and names the decode kernel only when the
  call is cached and one token long.
- **The prompt's kernel is the token loop.** The pure-torch
  `mamba_selective_scan` loops over tokens, rebinding `ssm_state` after each
  (`ssm_state_3`: the pscan and associative-scan branches bind `_0` and `_1`
  first, the zero init `_2`), and returns `(scan_output, ssm_state)`. The
  decode kernel updates once (`ssm_state_0`) and copies the result into the
  cache's buffer in place. So `STATE_OP = "ssm_state_3"`, `STEP_STATE_OP =
  "ssm_state_0"`; `_state_op(kernel)` picks the one for the kernel that
  fires, and `_token_state_op` and `_token_op` go through it.
  `route_kernels` binds each name to its own `torch_function`, and
  `needs_recurrent_routing` compares the chunk name with the chunk kernel's
  own function. With `use_mambapy` and `mambapy` installed the scan runs a
  parallel scan with no per-token binding; `needs_token_loop` reports it.
- **The arguments sit at different positions.** `x` is argument 0 of the
  scan and 1 of the decode kernel (the state is 0), and so on down to `C`;
  the bias is `delta_bias` in one and `dt_bias` in the other. `EProperty`'s
  `select` may be a function of the envoy, evaluated at read time after the
  key; `_argument(name)` looks the position up in `ARGUMENTS` by
  `_decoding(envoy)`.
- **The tensors are channel-first.** `x` and `dt` are `[batch, channels,
  seq]`, `B` and `C` `[batch, state_dim, seq]`, and a decode step drops the
  sequence axis. The preprocess returns `_tokens_first(value)` (a transpose,
  or an `unsqueeze(1)` on a decode step), a view, so in-place edits reach
  the kernel; the postprocess lays a written value back out
  (`_channels_first`). `B` and `C` get a `groups` axis of 1. The base's
  `_seq()` then reads the sequence off `attention_queries.shape[1]` as on
  the DeltaNet, and needs no override.

`betas` and `decays` are `DerivedEProperty`s: `softplus(dt + dt_bias)` and
`A * betas`, computed from the call's arguments (`_read(name)`) the same way
the kernel computes them, and read-only. `state_input` is `None` on a prompt
(the scan starts from zeros and takes no state; assigning raises) and a clone
of the decode kernel's argument 0. Two values live inside the kernels and
need the names bound to the pure-torch functions, not the dispatcher
(`needs_kernel_source`): `attention_head_outputs` at `READ_OPS` (`y` after
the `D` skip, before `silu(z)`: `scan_output_5` in the scan, `out_1` in the
decode kernel), and `state_output` on a decode step at `STEP_OUTPUT_OP`
(`ssm_state_to_0`, the state as it is copied into the cache). `state_output`
there cannot be `ssm_state_0`: `states` reads that binding on the same step,
and a location is served once. The consequence is a read order that differs
by kernel: in the scan `y` comes before the returned state, in the decode
step after the updated one.

The per-token reads of a decode step rest on `per_call`'s rule
([Once per call](#once-per-call-kernel-and-per_call)) on either mixer: a
step body that reads `state_input` first and `states` second computes this
step's `(op, first, seq)`, not step 0's, and so does one whose first read is
`linear_attn.input`.

## The state-space mixer: `StateSpace`

`StateSpace` (`nnterp/components/state_space.py`) is the Mamba-2 (SSD)
mixer on `mamba2`, `nemotron_h`, `bamba` and `falcon_h1`. Every one of these
modeling files carries the same copy of transformers' Mamba-2 code, so one
class serves them. The kernel choice is the base's: the forward decodes
through `mamba2_selective_state_update` when `use_precomputed_states and
seq_len == 1` and runs `mamba2_chunk_scan` otherwise, so a cached call over
several tokens takes the chunk scan, and `KERNEL` reads the binding
`use_precomputed_states_0` and the call's length off `SEQ_OP`
(`apply_mask_to_padding_states_0`, the first op after the binding on both
paths) once per call. The length is read inside the forward, not off the
mixer's `.input`, so a user's read of `linear_attn.input` earlier in the
step is not passed, and a step whose first read is `linear_attn.input`
decides its own kernel
(`tests/families/ssd.py::test_mixer_input_before_the_kernel_values`).

The forward differs from the DeltaNet's in four ways, and each is handled in
the subclass, not the base:

- **The kernels take their arguments in different places.** The scan is
  `(hidden_states, dt, A, B, C, chunk_size=, D=, dt_bias=, initial_states=)`,
  the update `(state, hidden_states, dt, A, B, C, D, dt_bias=)`.
  `CHUNK_ARGUMENTS` / `RECURRENT_ARGUMENTS` map each name to its position or
  keyword at the call site, and each value is declared with
  `select=argument(name)`: `EProperty.select` may be a function of the host,
  resolved at each access before the served read (so it may itself read an
  earlier value of the call, as the chunk scan's output select reads
  `return_final_states`).
- **The update has no sequence axis.** Its tensors are `[batch, heads,
  head_dim]` and `[batch, groups, state_dim]`, and `dt`, `A`, `D`, `dt_bias`
  are expanded over `head_dim` (and `state_dim` for `A`). The preprocess of
  each value adds a sequence axis of 1 on a decode step and the postprocess
  removes it. `betas` and `decays` are `EProperty` values on the kernel's
  `dt` argument (`select=argument("dt")`): the preprocess is the kernel's
  own arithmetic (`_gate`: plus `dt_bias`, softplus, the clamp to
  `dt_limit` on a prompt; times `A` for `decays`), taking element `[..., 0]`
  of the expanded `dt`, `dt_bias` and `A` on a decode step. The postprocess
  is the inverse (`_dt_for`): `log(expm1(b)) - dt_bias`, computed as `b +
  log(-expm1(-b))` so it does not overflow, without the clamp, expanded back
  over `head_dim` on a decode step; `decays` divides by `A` first.
- **The update returns only `y`.** It writes the new state into the cache's
  buffer (`state.copy_(ssm_states)`), so a decode step's `state_output` is
  read inside the kernel at `UPDATED_STATE = "ssm_states_0"`, the binding
  both the copy and the output read: a write there reaches the cache and
  `y`. On a prompt the scan returns `(y, final_state)` when the call has a
  cache and `y` alone without one, so `attention_head_outputs` selects `0`
  or the whole return and `state_output` raises `Unavailable` at the read
  without a cache.
- **Several values share one location, and some need another argument.**
  Seven values select from the kernel's `inputs`, and `betas`, `decays` and
  a prompt's `state_output` need arguments beyond their own (`dt_bias`, `A`,
  `return_final_states`), possibly after the model has moved into the
  kernel. `_arguments()` (`state_space.py:213-221`) reads the kernel call's
  `(args, kwargs)` once per call, `per_call(self, "arguments", lambda:
  Mediator.value(location))`, on the call's first need, and names them by
  the call site's table; a value that asks later in the call finds them in
  the record. The values themselves read and write through
  `Mediator.value` / `Mediator.swap` like any `EProperty`: nnsight answers
  a worker's repeated read of one location, and a read after a swap there,
  in the same visit, so the seven read together in one trace. A read of a
  location the model has moved past is an out-of-order read.

The state is `[batch, heads, head_dim, state_dim]` in the kernels; both
state values are served transposed, `[batch, heads, state_dim, head_dim]`,
so the shared `State` layout (key side first) holds, and transposed back on
assignment. `STATE_OP` is `None`: no kernel binds the state once per token,
and `route_kernels(family, "torch")` binds each kernel name to its own
`torch_function` (the dispatcher's closure is the one the DeltaNet kernels
use, `use_kernel_func_from_hub_with_fallback` over `mamba_ssm`; the routing
needed no change).

The per-token state comes from the chunk scan instead. Its inter-chunk
recurrence binds `new_states` (`CHUNK_STATES = "new_states_0"`), `[batch,
chunks + 1, heads, head_dim, state_dim]`: the state before each chunk and
after the last, computed in one cumulative step (a decay matrix over the
chunk boundaries times every chunk's contribution). `chunk_size` is the
mixer instance's attribute, read on every call, so `chunk_per_token(model)`
sets it to 1 on each `StateSpace` module of one model (recording the built
value in the module's `_nnterp_chunk_size` for `enabled=False`), and every
token is a boundary. `states` overrides the base's: a `DerivedEProperty`
that reads `CHUNK_STATES` through a module-level `EProperty`
(`_chunk_states`, keyed `source.<CHUNK_KERNEL>.source.new_states_0.output`,
so the read drills and orders like any value), after `_arguments` so the
record holds the arguments for later values; it serves `[:, 1:]`, transposed
and cloned. On a decode step it is `state_output` unsqueezed. Its predicate
is `needs_per_token_chunks` (the kernel reason, then the chunk size).
`state_after(t)` is `states[:, t]`. `state` and `set_state_after` are
`unavailable(...)` values (`NO_STATE_OCCURRENCES`, `NO_STATE_WRITES`), so
both are listed by `support()`: there is no per-token occurrence to walk, and
a boundary state written at token `t` would not flow into later boundaries,
which the same cumulative step computes from the chunk contributions and the
initial state, not from each other.

The tests are `tests/families/ssd.py`, mixed into each family's suite: the
shapes, the writes (`betas` and `decays` included, read back and checked
against the recurrence), the hand-off under `generate` with SSD's recurrence
checked on every decode step, the per-token `states` under `chunk_per_token`
checked against the recurrence token by token, values read together in one
trace, the `support()` reasons and the optimized-kernel reason.

Nemotron-H needs a block-level name choice, since each block holds one
`mixer` of four classes. Its `RENAME` keys the standard name on the mixer's
class (`{NemotronHMamba2Mixer: "linear_attn", NemotronHAttention:
"self_attn", NemotronHMoE: "mlp", NemotronHMLP: "mlp"}`): nnsight binds a
class key on every envoy that has exactly one direct child of that class, so
each block gets the name for what it holds and `support()`'s `_standard_children`
reads it like any alias.

## Adding a recurrent mixer

A new mixer subclasses `RecurrentMixer`, sets `CHUNK_KERNEL` and
`RECURRENT_KERNEL` (and `BRANCH` or `SEQ_OP` when its forward names those
bindings otherwise, or `KERNEL` itself, a `staticmethod`, when its forward's
test is another one), sets `STATE_OP` only when a kernel updates the
state once per token in a binding, and declares its values at
`kernel("inputs")` / `kernel("output")` with
`unavailable=needs_torch_kernels`; when the two kernels take an argument in
different places, `select` is a function of the host (`StateSpace`'s
`argument(name)`). `route_kernels`, the predicates,
`attention_output` and the per-token state come from the base; a kernel
whose tensors are not `[batch, seq, ...]` overrides `_seq`.

## Verification

`tests/families/test_qwen3_5_text.py` is the executable version of this
page, with `tests/test_base.py` for the base on its own.
`HF_HUB_OFFLINE=1 pytest tests/families/test_qwen3_5_text.py -q` passes 46
tests in about 13 s on CPU. The methods that pin the claims above:

| claim | test |
|---|---|
| the values follow the branch and the state hands off between steps | `test_values_follow_the_step_under_generate` (`:93-109`) |
| `state_output` is the state the last token leaves | `test_state_output_is_the_state_the_last_token_leaves` (`:79-86`) |
| `states` needs routing, and `support()` says so per block | `test_per_token_state_needs_the_recurrent_kernel` (`:111-116`) |
| `states[:, t] == state_after(t)`; a write flows into later tokens only | `test_per_token_state_with_the_recurrent_kernel` (`:118-148`) |
| `state` walks with the user's own `tracer.iter`; a first read pinned past 0 resolves | `test_state_iterates_with_the_users_own_iter` (`:150-178`) |
| under `generate`, the prompt is an inner loop on step 0 and each later step one token; a decode step's `states` is its own token | `test_per_token_state_within_a_generate` (`:180-205`) |
| two invokes reading one mixer each decide their own kernel, and nothing stays on the envoy | `tests/test_base.py::test_two_invokes_read_one_mixers_values` |
| a `model.edit` on a kernel value replays on another prompt | `tests/test_base.py::test_edit_on_a_kernel_value_replays_on_another_prompt` |
| several tokens over a cached state read the prompt's kernel | `tests/test_base.py::test_cached_call_with_several_tokens_reads_the_prompts_kernel` |
| a mixer with no `STATE_OP` reports `state`/`states` unavailable and still serves its kernel values | `tests/test_base.py::test_recurrent_mixer_without_a_state_op_reports_the_state_unavailable` |
| `route_kernels` `"torch"` / `"default"` round-trip the module's bindings, and `route_delta_rule` spells the same | `tests/test_base.py::test_route_kernels_round_trips_the_bindings` |
| with `STEP_STATE_OP` (Mamba-1) each name is bound to its own pure-torch function | `tests/test_base.py::test_route_kernels_binds_a_single_step_decode_kernel_to_its_own` |
| `select` given as a function of the host reads and writes | `tests/test_base.py::test_select_can_be_a_function_of_the_host` |
| the Mamba-1 values recompute the scan's own states and `y`; views take writes and in-place edits; the decode step, the per-token state and a `state_input` read before `states` in a step | `tests/families/scan_suite.py` (run by `test_mamba.py`, `test_falcon_mamba.py`, `test_jamba.py`) |

The canonical pattern above is the offset experiment in isolation; it ran
on `yujiepan/qwen3.5-tiny-random` with the printed `[0, 0, 1]`.

## Gotchas

- Call `route_kernels(model.family, "torch")` **before** the first trace
  that touches a recurrent layer of that family; a forward `.source` has
  already instrumented keeps the binding it was compiled with.
- Restore with `"default"` in a `finally` in tests: the routing is
  process-wide and the next test file's prompts would run the slow loop.
- `state_input` is a clone; edits to the read tensor do not reach the model,
  assignment does.
- A decode step's `states` offsets by earlier *decode* steps
  (`[0, 0, 1, 2, ...]`), not by the prompt: two locations, two counters.
- `states` reads every position: in a trace with a `set_state_after`, read
  it before the write.
- With `flash-linear-attention` or `causal-conv1d` installed every kernel
  value is unavailable; `support()` says so.

## Related

- [eproperty-internals.md](eproperty-internals.md) — function keys, `_resolve`'s walk and the once-per-access key
- [gotchas.md](gotchas.md)
- [testing.md](testing.md) — the hybrid test files
- nnsight `docs/usage/iter-all-next.md` — `tracer.iter` semantics; `docs/developing/interleaver-internals.md` — occurrences
