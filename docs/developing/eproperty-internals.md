---
title: EProperty Internals
one_liner: How nnterp's descriptors sit on nnsight's eproperty — one `EProperty` whose key is a path (`output`, `../norm.output`, `source.<op>.inputs`), availability, `_resolve`'s walk and the per-run drill into `.source`, `select`, the once-per-access key, the `Standard.sourced` flag, and derived values.
tags: [developing, internals, eproperty, source, descriptors]
related: [docs/developing/architecture.md, docs/developing/recurrent-mixer-internals.md, docs/developing/gotchas.md, docs/usage/availability.md]
sources: [nnterp/components/eproperty.py, nnterp/components/standard.py, nnterp/components/layer.py, nnterp/components/attention.py, nnterp/components/linear_attention.py, nnterp/components/recurrent.py, nnterp/standardized.py, nnterp/families/falcon.py, nnterp/families/llama4_text.py, nnsight src/nnsight/intervention/eproperty.py, nnsight src/nnsight/intervention/envoy.py, nnsight src/nnsight/intervention/source.py, nnsight src/nnsight/intervention/interleaver.py, nnsight src/nnsight/intervention/iterator.py, nnsight src/nnsight/intervention/util.py]
---

# EProperty Internals

## What this is for

Every standard value is a descriptor from `nnterp/components/eproperty.py`,
and every one of them is nnsight's `eproperty` (a `property` subclass over a
location string) plus two things nnsight does not have: an answer, before
anything runs, to "does this checkpoint have this value, and why not", and a
*path* for a key, so one descriptor serves a value wherever it lives: on the
host module, on a module named relative to it, or at an operation inside a
forward. This page is the contract of the descriptor and the nnsight facts
it depends on, with the line that establishes it on each side. Read nnsight
`docs/developing/extending-envoy.md` first if `eproperty`, location, and
`Mediator` are new words.

## Canonical pattern

A value of your own, located inside the shared attention forward, on a
Llama-family model (run on `hf-internal-testing/tiny-random-LlamaForCausalLM`):

```python
import torch
from jaxtyping import Float
from torch import Tensor
from transformers.models.llama.modeling_llama import LlamaAttention   # after `import nnterp`

import nnterp
from nnterp import StandardizedTransformer
from nnterp.components import EProperty, INTERFACE, interface_reason
from nnterp.families import llama


class Attention(llama.Attention):
    @EProperty(f"source.{INTERFACE}.source.repeat_kv_0.output", description="The keys after repeat_kv, [batch, heads, seq, head_dim]", unavailable=interface_reason)
    def expanded_keys(self, value: torch.Tensor) -> Float[Tensor, "batch heads seq head_dim"]:
        return value


model = StandardizedTransformer("meta-llama/Llama-3.1-8B", dispatch=True, attn_implementation="eager", envoys={LlamaAttention: Attention})
attn = model.layers[0].self_attn

Attention.expanded_keys.key                # 'source.attention_interface_1.source.repeat_kv_0.output'
Attention.expanded_keys.inside_forward()   # True
Attention.expanded_keys.dims               # ('batch', 'heads', 'seq', 'head_dim')
"(expanded_keys): The keys after repeat_kv" in repr(attn)   # True

with model.trace("Hello world there"):
    keys = attn.attention_keys.save()       # [batch, kv_heads, seq, head_dim], read first: repeat_kv fires later
    expanded = attn.expanded_keys.save()    # [batch, heads, seq, head_dim]
```

## The nnsight side, in six facts

1. **An `eproperty` is a `property` over `"{obj.path}.{key}"`.** Reading
   parks the worker with `Mediator.value(location)`, runs the decorated stub
   as the *preprocess* on the served value, and returns the result; writing
   runs `postprocess` and `Mediator.swap` (nnsight `eproperty.py:172-190`).
   `key` defaults to the stub's name (`:144-151`), and a `description` is
   only what the repr prints (`envoy.py:1082-1095`).
2. **A getter's `AttributeError` is swallowed.** A `property` raising
   `AttributeError` falls through to `Envoy.__getattr__` (`envoy.py:802-810`),
   which reports `'X' object (nor its module) has attribute 'name'`; the real
   error is lost (`eproperty.py:68-72`).
3. **`transform` is the write-back of a reshaping preprocess.** `__get__`
   binds a `WriteBack` (nnsight `eproperty.py`: the mapping
   `partial(transform, obj, view, raw)`, the view, and the eproperty,
   location and occurrence it was read at) as `mediator.transform`. It is a
   swap the worker has not issued yet: `Mediator.flush` issues it when the
   worker next moves on, on its next request or at the end of its block,
   tagged with the occurrence the view was read at. Its signature is
   `(self, view, raw)`: the edited view and the value as served. While it
   waits, a repeated read is answered with the view it holds, so
   `x.value[...] += f(x.value)`, which reads twice, edits the tensor that
   goes back. It fires whether or not the view was edited, so a transform
   must be an identity on an unedited view. nnterp's `EProperty.__get__`
   binds and checks the `WriteBack` the way nnsight's does.
4. **A location can be visited many times in one run**, and a worker asks
   for one *occurrence* of it (`Pending.iteration`, `interleaver.py:73-96`).
   `Mediator.iteration` is the occurrence the worker wants: `0` with no
   `tracer.iter`, an int a `tracer.iter[n]` pins, `None` when *relaxed*, which
   resolves to the mediator's own count of that location
   (`interleaver.py:161-167`, `:306-329`, `occurrence` `:331-334`). The first
   hit of a pinned non-zero step relaxes the mediator (`:478-483`).
   `Iterations.__iter__` pins before each step and restores the previous pin
   on exit (`iterator.py:121-144`).
5. **An operation inside a called function exists only after a drill in the
   current run.** `SourceEnvoy.source` writes a `None` placeholder into
   `interleaver.sourced`, parks on `{path}.fn`, receives the live callee from
   `run_op` and stores the instrumented copy back (`source.py:788-833`,
   `run_op` `:495-503`); `interleaver.sourced` is cleared on every run's entry
   (`interleaver.py:710`). The `.fn` handoff happens *before* the call runs,
   and a `Source.__getattr__` for an unknown name is an `AttributeError`
   listing the available ops (`source.py:1014-1033`). A module's own
   `.source` is different: `Envoy.source` instruments the forward once, for
   good, and works outside a trace (`envoy.py:645-663`).
6. **The tree is navigable downward from any envoy, and an envoy knows its
   own path but not its parent.** `Envoy.get(path)` resolves a dotted path
   from an envoy, aliases included (`envoy.py:954-975`); `.path` is the
   native dotted name from the root (`model.model.layers.0.self_attn`). An envoy's `.input` is the call's first argument,
   `first_input` over the served `(args, kwargs)` pair, and assigning it is
   `replace_first_input` (`envoy.py:569-584`, `util.py:20-36`).

Everything below is a consequence of these six.

## `EProperty`: one descriptor, a path for a key

`EProperty(eproperty)` (`nnterp/components/eproperty.py:35-259`) takes
`key`, `description`, `unavailable` and `select` (`:86-96`). A string `key`
is stored as nnsight's `key`; a callable one is stored as `locate` and the
`key` shown in the repr and in `.key` is `<its name>` (`<kernel.inputs>`,
`<apply_rotary_pos_emb_0|query_layer_0>`, `<_token_state_op>`).
`__set_name__` (`:98-104`) fills `name` and `key` from the class body when
they are still `None`: a bare marker `attention_probabilities =
unavailable("...")` (`:262-269`) is never called on a stub, so nnsight's
`__call__` never ran to set them; without this the marker would have no
name in the repr and `support()`.

### The path grammar

A key is dotted segments ending in `output`, `input` or `inputs`, walked
from the host envoy, or from the model's root after a leading `/`:

| segment | meaning | example |
|---|---|---|
| `output` (last) | the current node's output | `"output"`: `Layer.layer_output` (`layer.py:58`), a view over the same location as `.output` |
| `input` (last) | the first argument of the current node's call (fact 6) | `"source.dropout_add_0.input"`: BLOOM's contribution (`families/bloom.py:73-78`) |
| `inputs` (last) | the `(args, kwargs)` pair, one element of it with `select` | `"inputs"` on the root: `input_ids` (`standardized.py:352-360`); `"source.attention_interface_1.inputs"` with `select=1`: `attention_queries` (`attention.py:140`) |
| `/` (leading) | the model's root (`envoy.root`), walked down from there like a path below the host, aliases included | `"/projector.output"`: `Vision.image_features` (`components/vision.py`); `"/inputs"`: `Vision.image_token_mask` |
| `../` (leading, repeatable) | the parent module, by native name | `"../post_attention_layernorm.output"`: Gemma-2's `attention_output` (`families/gemma2.py:30-36`) |
| a name | a child module of the current node, aliases included; under a `source`, an operation | `"embed_tokens.output"`: `token_embeddings` (`standardized.py:186-194`) |
| `source` | the current module's or operation's forward, instrumented for this run | `"source.attention_interface_1.source.nn_functional_dropout_0.output"`: `attention_probabilities` (`attention.py:196-202`); `"../source.hidden_states_view_0.output"`: Llama 4's `mlp_output` (`families/llama4_text.py:106-112`) |
| a function of the host | returns a path, at read time, inside the trace | Falcon's `by_alibi(without, with_alibi, attribute)` (`families/falcon.py:51-58`), a `RecurrentMixer`'s `kernel("inputs")` (`recurrent.py:247-254`), which names the call `RecurrentMixer.KERNEL` picks (`:311-329`) |

### `_resolve`: the walk

`_resolve(obj, key)` (`:189-222`) turns a path into the served location, and
runs before every read and write, because of fact 5: an operation under a
call is per run, so nothing about the walk can be cached on the descriptor.
A leading `/` swaps the host for `obj.root` (nnsight's `Envoy.root`, the
top of the parent links) and is dropped; the rest is then a relative path
from the root, and a `../` right after it is refused, since nothing is above
the root. It strips every leading `../` (counting them), splits the rest on `.`, and,
below the host, hands every segment but the last to one
`obj.get(".".join(walk))` (fact 6): every segment is an attribute, whether
a child module (so an alias works), `source` (the drill of fact 5) or, on
the `Source` a drill returns, an operation. An
`AttributeError` anywhere is re-raised as `SourceNotAvailable` naming the
value, the path and nnsight's list of what is there (`:216-220`), because of
fact 2. The key it is given is `path(obj)` (`:176-178`: the string, or the
function applied to the host). Verified on tiny GPT-2: a wrong op
reads `model.transformer.h.0.attn.broken reads 'source.no_such_op_0.output',
which this run does not have: 'model.transformer.h.0.attn.source' has no
operation 'no_such_op_0'; available: is_cross_attention_0, ...`.

The result is `f"{node.path}.{attribute}"` with `inputs` folded into `input`: `input` and `inputs` are the same served location, the pair;
picking the first argument out of it is `_pick`'s job. So a value at any
depth is served by `Mediator` at a location string, the same way a plain
eproperty's is.

### Above the host: arithmetic on names

A path that starts with `../` is not walked through envoys: `_resolve` drops as many trailing
segments from the host's own path as there are `../`, appends the rest of
the key, and returns that string as the location. The parent's name is
native, so a `../` step is by native name, and the segment after it is
joined as written, so a sibling is named by its native name too
(`"../post_attention_layernorm.output"` on `model.model.layers.0.self_attn`
is `model.model.layers.0.post_attention_layernorm.output`). Nothing is
drilled: a `source` segment above the host names an operation of the
parent's own forward, which is served by its string because the family
instrumented that forward at build (`sourced = True`, below); a second
`source` on such a path, a call inside the parent's forward, would need the
parent drilled at read time, and `_resolve` refuses it with a `ValueError`
saying so. A value that reads something far from its host by standard name
anchors its key at the root (`/`) instead, which is walked through envoys. Verified on tiny GPT-2: `EProperty("../ln_2.output")`
on the attention equals `layers[0].ln_2.output`, and zeroing it in place
moves the logits.

### The drill and a `tracer.iter` pin

A `source` segment on the walk is a plain `node.source`, and `EProperty`
knows nothing of pins: the drill and the read that follows both ask for the
occurrence the worker's mediator names. On the module envoy, or on an
operation already drilled this run (its path is in `interleaver.sourced`),
nothing is asked of the model. The first drill of a run into a call is fact
4 meeting fact 5: the drill parks on `{path}.fn`, a location the call fires
**once per forward**. Inside `tracer.iter[7]` the mediator is pinned to
occurrence 7, so where the pin is a token index inside one call the `.fn`
read would wait for the eighth firing of a call that fires once, and the
run would end with it parked. A value pinned that way relaxes the pin
itself, in its key function: `RecurrentMixer._token_state_op`
(`recurrent.py:352-362`) decides the call's kernel and drills into the
kernel call inside `with pinned(None):` (`:231-244`, which sets the
mediator's `iteration` and restores it on the way out), so the drill
resolves to the call in flight, and then returns the key; the walk finds
the call drilled and the value read that follows is pinned to the token.
This is what lets `for t in tracer.iter[2]: mix.state` resolve on a first
read pinned past 0 (`tests/families/test_qwen3_5_text.py:173-176`).

### `_pick` / `_put`: `select`, and why `input` is the first argument

`_pick(attribute, value)` (`:199-207`) takes one element of the served
value and `_put(attribute, current, element)` (`:209-223`) puts one back;
`attribute` is the path's last segment past any `/` (`EProperty.attribute(key)`),
handed in by the caller:

- `input` is `first_input(args, kwargs)` on read and
  `replace_first_input(args, kwargs, element)` on write, nnsight's own rule
  for `.input` with nnsight's own helpers (fact 6), so `EProperty("ln_2.input")`
  on a block reads what `block.ln_2.input` reads. Verified on tiny GPT-2: it
  equals `layer.input + attention_output`, and assigning it moves the logits
  with the call's other arguments intact. Nothing is destructured in the
  stub.
- `inputs` with an int `select` is `args[n]`, with a str `kwargs[name]`
  (`attention_queries` is `select=1`, `attention.py:140`; a DeltaNet `decays`
  is `select="g"`, `linear_attention.py:66`); with no `select` it is the
  pair (the root's `input_ids`, whose stub takes `kwargs["input_ids"]` and
  whose postprocess puts it back, `standardized.py:352-360`).
- `output` with an int `select` is one element of the returned tuple
  (`attention_head_outputs` is `select=0`, `attention.py:215`); with no
  `select` it is the value as returned.

A write that selects (a `select`, or an `input` attribute, `:257`)
re-reads the current value with `Mediator.value`, puts the element in and
swaps the whole back (`__set__`, `:249-259`), which is why assigning
`attention_keys` swaps in the keys and keeps every other argument.

### `__get__` / `__set__`: nnsight's machinery on every path

`__get__` (`:230-247`) is `_check`, one `path(obj)`, `_resolve` over it,
`Mediator.value`, `_pick` with the key's last segment, the preprocess, and,
when a `transform` is registered, the bind of fact 3 with the raw served
value riding along (`:241-246`). `__set__` (`:249-259`) is `_check`, the
postprocess, one `path(obj)`, `_resolve`, `_put` when selecting, and
`Mediator.swap`. Both call `Mediator` directly; a host has no hook in
between. The key is computed **once per access** (`:226-228`) and
its last segment passed down, never re-derived: a key function may read a
served value pinned to the current step (`RecurrentMixer.KERNEL` reads the
forward's bindings as a step's first, pinned read, fact 4), and a second
evaluation after that read would run with the pin relaxed and ask the model
for an occurrence it has moved past. Because the location is served by
`Mediator` whatever the path, `preprocess`, `postprocess` and `transform`
all work on every value, including an operation inside a forward. Families that serve a transposed
view of head outputs (`seq_first`, `attention.py:48-57`;
`families/mpt.py:65-71`) rely on a transpose being a view of the same
storage, so in-place edits land, and transpose back in `postprocess`; a
value that needs a copy carries edits back with a transform (Falcon's
`mlp_output`, `families/falcon.py:136-148`), and the transform's `raw` is
what a tuple-returning module would need to rebuild its container
(nnsight `eproperty.py:52-66`).

### Introspection

- `path(obj)` (`:154-156`): the key for this host, the string or what the
  function returns.
- `inside_forward()` (`:158-160`): a `source` segment on the key,
  leading `../` stripped first, or any function key (every function key in
  the tree names an operation). So `"source.dropout_add_0.input"` and
  `"../source.hidden_states_view_0.output"` both answer `True`, and
  `"../post_attention_layernorm.output"` `False`. The suite uses it to pick
  the interior values of a family's `Attention` (`tests/families/suite.py:320-335`).
  It says where a value is, not when its forward has to be instrumented;
  that is the host's `sourced` flag, below.

## Availability

`unavailable` is a reason string, or a predicate `f(envoy) -> str | None`.
`reason(obj)` evaluates it on the *instance* (`:108-110`), so a checkpoint's
config decides (`needs_eager`, `components/attention.py:18-23`, reads
`envoy._module.config._attn_implementation`) and so a hybrid can answer per
block.

- `_check` (`:112-123`) runs before every read and write. A reason raises
  `Unavailable` (a `RuntimeError`, `:21-32`) naming `{obj.path}.{name}` and
  the reason. A predicate that itself raises `AttributeError` is re-raised as
  `RuntimeError("the availability check of ... failed: ...")` (`:115-121`),
  because of fact 2: left alone, a typo in a predicate would surface as "no
  attribute `attention_probabilities`" and hide the predicate's own bug.
  Verified: a predicate reading `config.no_such_flag` raises
  `RuntimeError: the availability check of model.transformer.h.0.attn.probe failed: 'GPT2Config' object has no attribute 'no_such_flag'`.
- `layout` and `dims` (`:127-150`) read the return annotation of the stub with
  `typing.get_type_hints(func, include_extras=True)`. The annotation is one
  of the thirty-five layout aliases, each defined in the file of the envoy that
  serves it (`Residual = Float[Tensor, "batch seq hidden"]` in `layer.py`,
  `Pattern` and `Keys` in `attention.py`, `State` in `recurrent.py`,
  `Logits` in `standardized.py`):
  `get_type_hints` evaluates the string annotation (the components use
  `from __future__ import annotations`) in the stub's module globals, where
  the alias is imported, and returns the alias object itself, so
  `Attention.attention_keys.layout is Keys` and a family's redefinition
  annotated `-> Keys` has the identical layout. An inline `Float[Tensor, "..."]`
  resolves the same way (`expanded_keys` above). A `State | None` annotation
  (`LinearAttention.state_input`, which is `None` on a fresh prompt) is a
  `types.UnionType`, so the non-`None` member is taken (`:142-143`). `dims` is
  `layout.dim_str.split()`. Verified:
  `LinearAttention.state_input.layout is State` and
  `LinearAttention.state_input.dims == ('batch', 'heads', 'key_dim', 'value_dim')`.
- `hasattr(envoy, "value")` and `getattr(envoy, "value", None)` both **raise**
  `Unavailable`, since Python's default only swallows `AttributeError`
  (`tests/test_base.py:49-52`). The reason this is not turned into an
  `AttributeError` is fact 2 again: the reason text would be lost. Use
  `support()`.

## `Standard`: the `sourced` flag

`Standard` (`components/standard.py:61-96`) is the envoy every component
derives from, and it carries one flag about forwards. A value that is an
operation inside a forward is served only on a call whose forward was
instrumented before the call began. The drill at read time (fact 5) is
enough when the read comes before the module runs, the usual case: a value
on the host's own `source.` is the first request on that module or it is
out of order, as with a bare `.source`
([finding-source-ops.md](../extending/finding-source-ops.md)). It is too
late when the worker is already parked inside the module, which a read can
legitimately do: Llama 4's block is read as `attention_output` (the
attention module's output, after the block is running) and then
`mlp_output` (`../source.hidden_states_view_0.output`, an operation in the
block's forward). The `.fn` handoff and the instrumented forward are both
installed before a call, so the first trace that reads the two in that
order ends with an `OutOfOrderError` on the view. Nothing infers this from
a path; the family that puts a value in such a forward is the one that
knows, and it says so on the envoy that owns the forward: `sourced = True`
on its class (`sourced = False` on `Standard`, `:50`). `__init__` (`:52-55`)
and `_update` (`:57-60`) then touch `self.source`, at build and again when
real weights replace meta ones and nnsight reinstalls the plain forward
(nnsight `envoy.py:391`). Llama 4's `Layer` is the one instance
(`families/llama4_text.py:78-86`): `sourced = True` under a docstring
saying why, its `Mlp.mlp_output` being an operation in the block's forward
read after the block's attention has returned. Verified on tiny GPT-2: a
custom `Mlp` with `EProperty("../source.hidden_states_1.output")` passed
through `envoys=` and read after `attention_output` ends with
`OutOfOrderError: 'model.transformer.h.0.source.hidden_states_1.output.i0'
was requested but the model already ran past it`; with `class
Layer(gpt2.Layer): sourced = True` passed beside it, the read is served.
Only the modules a family flags are instrumented, so the per-forward cost
of `.source` stays where a family asked for it. `EProperty` has no part in
the flag: `inside_forward()` says where a value is, not when its forward
is instrumented.

`values()` (`:62-65`) collects a class's `EProperty`s by name, base classes
first, through the module-level `values(cls)` (`:24-31`) the root's
`support()` uses for its own class, and `support()` (`:67-69`) maps each to
its reason on this instance.

## `DerivedEProperty`: computed, read-only

`DerivedEProperty(EProperty)` (`eproperty.py:272-292`) has no location: `__get__`
(`:285-289`) runs `_check` then `compute(obj)`, which may read any number of
served values in forward order; `__set__` raises `AttributeError("... is
derived and read-only")` (`:291-292`), which is the one place an
`AttributeError` is right (it is the assignment that fails). `_preprocess`
is set to `compute` (`:283`) **only** so that `layout` reads its return
annotation; it is never called as a preprocess. `RecurrentMixer.states`
(`recurrent.py:418-422`) is the one instance.

A forward that branches names its op with a function key decided once per
module call: `RecurrentMixer.KERNEL`, `per_call` and `pinned` live in
`nnterp/components/recurrent.py`, and
[recurrent-mixer-internals.md](recurrent-mixer-internals.md#once-per-call-kernel-and-per_call)
describes them.

## Where it lives

| value | key | location it serves | write path |
|---|---|---|---|
| boundary value (`layer_output`, `attention_output`, `logits`) | `EProperty(key="output")` | `{path}.output` (same as `.output`) | `postprocess` → swap; `transform` available |
| the root's call arguments (`input_ids`, `attention_mask`, `input_size`) | `EProperty(key="inputs")` | `{path}.input`, the `(args, kwargs)` pair | `postprocess` repacks the pair → swap |
| another module's value (Gemma-2 `attention_output`, `token_embeddings`) | `EProperty("../norm.output")`, `EProperty("embed_tokens.output")` | that module's `.output` | as any value |
| value inside a forward (`attention_probabilities`, `decays`) | `EProperty("source.<op>.<attribute>", select=...)` | `{path}.source.<op>.{input\|output}` after a drill | `postprocess` → `_put` → swap |
| an operation in the parent's forward (Llama 4 `mlp_output`) | `EProperty("../source.<op>.output")` | the parent's `.source.<op>.output`; the parent's class sets `sourced = True` | as any value |
| computed (`states`) | `DerivedEProperty(compute)` | none | refused |
| declared missing | `unavailable("reason")` | none | refused with the reason |

One thing on the root is none of these. A size (`num_layers`, `hidden_size`,
`vocab_size`, `num_heads`, `num_kv_heads`, `head_dim`, `qk_head_dim`,
`intermediate_size`) is a `StandardizedProperty` (`nnterp/standardized.py:26-50`),
a bare descriptor with no `eproperty` underneath: no location, nothing served
during a trace, no `description`, `layout` or `support()` entry.
Its `__get__` (`:43-47`) does one lookup, `getattr(obj.family, name)`, and calls
that function with the model when the family module defines it, else the
wrapped plain rule (`:399-433`); its `__set__` (`:49-50`) raises
`AttributeError("<name> is read off the config; a family defines `def <name>(model)`
to say it otherwise")`, so an assignment cannot shadow it. It is what lets `falcon.py` say
`num_kv_heads` and `deepseek_v2.py` say `head_dim` without a subclass of the root.

## Gotchas

- `hasattr` / `getattr(..., default)` raise `Unavailable`; only
  `AttributeError` counts as absence in Python, and turning `Unavailable`
  into one would lose the reason through `Envoy.__getattr__`.
- A nested `.source` (an operation's) only works inside a trace, and the drill
  parks *before* the call fires; a value read after the op's `.output` cannot
  then drill into it (nnsight `source.py:807-813`, the `.fn` ordering).
- `interleaver.sourced` is cleared on entry to every run, so nothing about a
  drill can be cached on the descriptor across traces; `_resolve` walks the
  path on every access by design.
- An out-of-order read of a value inside a call the forward makes raises
  `OutOfOrderError` naming the call's `.fn`, the location the drill waits
  on, not the value's own (`'...attention_interface_1.fn.i0' was requested
  but the model already ran past it`). See [gotchas.md](gotchas.md).
- A value read after its forward has started (Llama 4's `mlp_output`, a
  `../source.` key read after `attention_output`) needs that forward
  instrumented ahead of the trace, and nothing infers it from the path: the
  envoy that owns the forward sets `sourced = True`, or the read is an
  `OutOfOrderError`.
- `input` is the first argument, never the pair; a stub on an `input` path
  receives a tensor. The pair is `inputs`.
- `select` on `output` indexes a tuple; on a module that returns a bare
  tensor use no `select`.
- `.source` instruments the forward over a snapshot of the module's globals
  at first drill (nnsight `source.py:439-471`, `function_like` copies
  `fn.__globals__`); a kernel binding switched after that is not seen. See
  `route_kernels` in [recurrent-mixer-internals.md](recurrent-mixer-internals.md).

## Related

- [architecture.md](architecture.md) — where the descriptors sit in the tree
- [recurrent-mixer-internals.md](recurrent-mixer-internals.md) — function keys in use: `RecurrentMixer.KERNEL`, `per_call`, `pinned`
- [gotchas.md](gotchas.md)
- nnsight `docs/developing/extending-envoy.md`, `docs/developing/source-internals.md`, `docs/developing/interleaver-internals.md`
