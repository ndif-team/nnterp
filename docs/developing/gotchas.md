---
title: Developing Gotchas
one_liner: The traps a contributor meets writing families, descriptors and tests — each as the constraint and the reason, with where it comes from.
tags: [developing, gotchas, internals, source, tracing]
related: [docs/developing/eproperty-internals.md, docs/developing/recurrent-mixer-internals.md, docs/developing/testing.md, docs/developing/architecture.md, docs/usage/availability.md]
sources: [nnterp/components/eproperty.py, nnterp/components/attention.py, nnterp/components/linear_attention.py, nnterp/components/recurrent.py, nnterp/families/falcon.py, nnterp/families/gpt2.py, tests/families/suite.py, nnsight src/nnsight/intervention/envoy.py, nnsight src/nnsight/intervention/interleaver.py, nnsight src/nnsight/intervention/source.py]
---

# Developing Gotchas

## What this is for

Each entry is a constraint a contributor runs into, why it holds, and the
line it comes from. User-facing versions of some live under `docs/usage/`;
these are the ones you meet writing a family, a descriptor or a test. The
examples are on `hf-internal-testing/tiny-random-gpt2` and
`hf-internal-testing/tiny-random-LlamaForCausalLM` unless said otherwise. The
tracing rules these entries rest on are nnsight's, documented in the nnsight
repository (https://github.com/ndif-team/nnsight, `docs/`) and in the `nnsight`
plugin of https://github.com/ndif-team/skills.

## Canonical pattern

The shape most entries take: bind outside, read in forward order, save what
you keep.

```python
import nnterp
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", dispatch=True, attn_implementation="eager")
read = {}                                              # bound outside the block
with model.trace("Hello world there"):
    for layer in model.layers:                         # forward order: block 0 before block 1
        read[layer.path, "pattern"] = layer.self_attn.attention_probabilities.save()   # inside the block ...
        read[layer.path, "resid"] = layer.layer_output.save()                           # ... before it returns
```

## Tracing

**Block-local names do not survive a trace.** `marker = 42` inside `with
model.trace(...)` is gone after the block; only `.save()`d values are pushed
back, and a plain list or dict built inside needs `.save()` too. The suite
pre-binds its containers
(`FamilySuite.test_every_value_reads_a_tensor_on_every_layer`,
`FamilySuite.test_interface_reproduces_heads_and_pattern`,
`tests/families/suite.py`).

**Reads follow the forward within one invoke.** Reading block 3 then block
1 raises `OutOfOrderError` ('model.transformer.h.1.output.i0' was requested
but the model already ran past it). Falcon binds values before the rotary
that produces queries and keys, so read `attention_values` first
(`Attention`, `nnterp/families/falcon.py`;
`TestFalcon.test_values_bind_before_the_rotary`,
`tests/families/test_falcon.py`); on DeltaNet, `states` reads every
position, so it goes before any state write
(`nnterp/components/recurrent.py`).

**An out-of-order read of a *source-located* value names the drill, not
the value.** A value inside a call the forward makes is reached by drilling
into that call, and the drill parks on `{op}.fn`
(`EProperty._resolve`, `components/eproperty.py`), which fires before the
call runs.
`layer_output` then `attention_probabilities` of the same block
raises `OutOfOrderError: '...attention_interface_1.fn.i0' was requested but
the model already ran past it`, and no save of the block is bound. Read in
forward order.

**Nested `.source` only inside a trace.** `envoy.source` works anywhere and
instruments the module; `op.source` needs the live callee and raises
`SourceNotAvailable("recursive .source is only available inside a trace")`
outside one (nnsight `SourceEnvoy.source`, `source.py`). The `.fn` handoff
also fires *before* the call runs, so drill before reading that op's
`.output`.

**`interleaver.sourced` is cleared per run.** Every drill is redone on
every trace (`EProperty._resolve`, `components/eproperty.py`); never cache
a `SourceEnvoy` on a descriptor or a family. A drilled op is reused within
one run, across generation steps (nnsight `run_op`, `source.py`;
`Interleaver.sourced`, `interleaver.py`).

**A served location is re-read only while the worker is parked there.**
`_token_op` reads `_seq()` (the queries, on DeltaNet) first *because* that read
parks the worker at the kernel call's start, the one moment the state op's
count is what earlier calls put through it (`recurrent.py`). A value computed from
several served reads has to think about *when* each read happens.

**Occurrence indices are absolute per location over the run.** The
interleaver counts every visit for the life of the tree, and a worker's own
occurrence is that count minus the count when it started (nnsight
`Interleaver.counts`, `Mediator.occurrence`, `interleaver.py`). A decode
step's one token is occurrence `k - 1` of the recurrent op on step `k`, so
`states`, `state_after` and `set_state_after` offset by `_token_op()`'s
`first` ([recurrent-mixer-internals.md](recurrent-mixer-internals.md)).

## Descriptors

**`hasattr(envoy, value)` raises `Unavailable`.** Only `AttributeError`
counts as absence, and `Unavailable` is a `RuntimeError` on purpose: an
`AttributeError` from a descriptor is rewritten by `Envoy.__getattr__` and
the reason is lost (`EProperty._check`, `components/eproperty.py`; nnsight
`envoy.py`). Use `support()`.

**`getattr(envoy, name, None)` inside a trace.** The default only covers
`AttributeError`, so `Unavailable` passes through, and reading an available
value parks the worker and spends the read. Decide structure outside the
trace; inside, ask `support()`.

**`.source` snapshots module globals.** The instrumented forward is a new
function built over the original's `__globals__` (nnsight `function_like`,
`instrument`, `source.py`) and cached per code object; a module-level name
rebound afterwards (a kernel switch, a monkeypatch) is not what the
instrumented copy sees. `route_kernels` must therefore run before the first
trace of that layer (`recurrent.py`).

**`.source` on a vLLM transformers-backend module sees the class's text.**
The backend rewrites the forward of a module it fuses (`q_proj`, `k_proj`,
`v_proj` into `qkv_proj`; a gated MLP's into `gate_up_proj`) as an AST,
compiles it under the class's file and line numbers and binds it on the
instance (vLLM `models/transformers/fx_utils.py` `compile_forward`,
`fusers/base.py` `fuse`). `inspect.getsource` on that function returns the
file at those lines: the class's forward, which calls the projections the
fusion removed. So a value keyed on `source.<op>` of such a module (the
queries and keys before the rotary, `apply_rotary_pos_emb_0`) cannot be
served there; `transformers_backend.Attention.no_rotary` refuses with
`BACKEND_ROTARY`. Values on the module's children (`qkv_proj`, `o_proj`, the
mounted `attn`) are hooks and unaffected.

**A key function stored on a class is a bound method through an instance.**
`RecurrentMixer.KERNEL` is a `staticmethod`
(`recurrent.py`), so `self.KERNEL(self)` and `type(self).KERNEL(self)`
are the same call. A subclass that sets `KERNEL` itself wraps it in
`staticmethod` too, or reaches it through the class; a bare function reached
through an instance passes the envoy twice and raises.

**An `EProperty` is not cloudpicklable by value.** `pickle.dumps(nnterp.Layer.layer_output)`
is `TypeError: cannot pickle 'EProperty' object`. A family that travels to
NDIF travels by reference (`StandardizedTransformer._remoteable_class`,
`standardized.py`); do not `nnsight.register(nnterp)` by value.

**`envoys=` matches type before path.** `_resolve_envoy_class` walks the
module's MRO against type keys, then path-suffix keys against the native path
and every alias spelling of that path (nnsight
`Envoy._resolve_envoy_class`). On GPT-2 `envoys={"self_attn": Marker}` and
`envoys={"attn": Marker}` both do nothing, because the family's
`GPT2Attention` type key is tried first. Key on the type to displace
(`test_user_envoys_merge_over_defaults`, `tests/test_registry.py`). Alias
paths compose through ancestors, so under `transformer.h -> layers` GPT-2's
`transformer.h.0` is also `layers.0`, and `"layers.*"` (a `*` is any one
component) reaches every block; the default family keys its `Layer` that
way.

**A bare `unavailable("...")` needs `__set_name__`.** The marker is never
called on a stub, so nnsight's `eproperty.__call__` never set `name`/`key`;
`EProperty.__set_name__` (`components/eproperty.py`) fills them from
the class body. A descriptor subclass that overrides `__set_name__` must keep
that.

## In-place edits

**GPT-2's and GPT-BigCode's queries, and MPT's queries, keys and values, are
split views.** torch refuses `attention_queries[:, 0] = 0` ("Output 0 of Select
is a view and is being modified inplace"); assign a new tensor instead
(`tests/families/suite.py`, `test_interior_in_place_edits` with
`REFUSES_IN_PLACE_QKV`).

**Falcon adds the attention into the MLP's output tensor in place.** A live
`mlp.output` would read as `mlp + attn` by the end of the block, so
`mlp_output` reads a clone and an `@mlp_output.transform` carries in-place
edits on the clone back (`families/falcon.py`;
`TestFalcon.test_mlp_output_is_a_copy_the_block_does_not_touch`,
`TestFalcon.test_in_place_mlp_edit_reaches_the_model_through_the_transform`,
`test_falcon.py`). Any new value on Falcon's block has to ask whether the
block will later write into its tensor.

**DeltaNet's cache buffer is overwritten in place.** The kernel receives
the cache's own tensor as `initial_state` and the cache then writes the new
state into it; `state_input` returns a clone (`linear_attention.py`).

## Related

- [eproperty-internals.md](eproperty-internals.md)
- [recurrent-mixer-internals.md](recurrent-mixer-internals.md)
- [testing.md](testing.md)
- [transformers-compat.md](transformers-compat.md)
- nnsight `docs/errors/index.md`, `docs/errors/out-of-order-error.md`
