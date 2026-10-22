---
title: Finding Source Ops
one_liner: Discover the operation names a `source.` path needs — `print(envoy.source)` outside a trace, `<call>.source` inside one, and how nnsight names calls and bindings.
tags: [extending, source, operations, transformers]
related: [docs/extending/overriding-values.md, docs/extending/custom-values.md, docs/extending/adding-a-family.md]
sources: [nnterp/components/eproperty.py, nnterp/components/attention.py, nnterp/families/gptj.py, nnterp/families/bloom.py, nnterp/families/falcon.py, nnterp/families/gpt_oss.py]
---

# Finding Source Ops

## What this is for

A source-located value names an operation inside a module's forward:
`attention_interface_1.source.nn_functional_dropout_0` for the pattern, `dropout_add_0`
for BLOOM's contribution, `self__attn_0` for GPT-J's attention call. Those names come
from transformers' own source, as nnsight labels it, and the only correct list is the
one printed from the version you run. This page is how to read that list, what the
labels mean, and which one to pick. nnsight docs/usage/source.md is the full account of
`.source`; this is what a family author needs from it.

## Canonical pattern

Outside a trace, print the module's source; the labels on the left are the operation
names. Trimmed from `hf-internal-testing/tiny-random-gpt2` on transformers 5.17:

```python
import nnterp
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", attn_implementation="eager")
print(model.layers[0].self_attn.source)
```

```
                                             * def forward(
 ...
 self_c_attn_1                           -> 40         query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
 split_1                                 ->  +         ...
 ...
 query_states_view_0                     -> 46     query_states = query_states.view(shape_q).transpose(1, 2)
 transpose_4                             ->  +     ...
 query_states_1                          ->  +     ...
 ...
 using_eager_0                           -> 56     using_eager = self.config._attn_implementation == "eager"
 ALL_ATTENTION_FUNCTIONS_get_interface_0 -> 57     attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
 attention_interface_0                   ->  +     ...
                                            61     if using_eager and self.reorder_and_upcast_attn:
 self__upcast_and_reordered_attn_0       -> 62         attn_output, attn_weights = self._upcast_and_reordered_attn(
                                            65     else:
 attention_interface_1                   -> 66         attn_output, attn_weights = attention_interface(
                                            67             self,
                                            68             query_states,
                                            69             key_states,
                                            70             value_states,
                                            71             attention_mask,
 ...
 self_c_proj_0                           -> 78     attn_output = self.c_proj(attn_output)
 attn_output_1                           ->  +     ...
 self_resid_dropout_0                    -> 79     attn_output = self.resid_dropout(attn_output)
 attn_output_2                           ->  +     ...
                                            81     return attn_output, attn_weights
```

The interface call is `attention_interface_1` on every family that uses it, which is why
`nnterp.components.INTERFACE` is that string. Its arguments are `(self, query, key, value,
attention_mask, ...)`, so the base `Attention` reads the queries as
`EProperty("source.attention_interface_1.inputs", select=1)`.

## Inside a call: nested `.source`, inside a trace only

`attention_interface_1` calls a function, transformers' `eager_attention_forward`. Its
operations exist only under `attention_interface_1.source`, and that drill works only
inside a trace: the callee is a local variable bound at run time, so nnsight resolves it
from the live call (`.fn`) the first time someone drills into it in that run, and clears
what it built at the start of the next. Outside a trace,
`attn.source.attention_interface_1.source` raises
`SourceNotAvailable: recursive .source is only available inside a trace`.

Trimmed from `hf-internal-testing/tiny-random-LlamaForCausalLM`, read inside a trace:

```python
model = StandardizedTransformer("meta-llama/Llama-3.1-8B", attn_implementation="eager")
attn = model.layers[0].self_attn
with model.trace("The Eiffel Tower is in"):
    args = attn.source.attention_interface_1.inputs.save()      # the call's arguments: served first
    inner = attn.source.attention_interface_1.source            # the drill: served next, from the live callee
    print(inner)
    probs = inner.nn_functional_dropout_0.output.save()         # then an op inside the body
```

```
                             * def eager_attention_forward(
 repeat_kv_0             ->  9     key_states = repeat_kv(key, module.num_key_value_groups)
 key_states_0            ->  +     ...
 repeat_kv_1             -> 10     value_states = repeat_kv(value, module.num_key_value_groups)
 value_states_0          ->  +     ...
 key_states_transpose_0  -> 12     attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
 torch_matmul_0          ->  +     ...
 attn_weights_0          ->  +     ...
                            13     if attention_mask is not None:
 attn_weights_1          -> 14         attn_weights = attn_weights + attention_mask
 nn_functional_softmax_0 -> 16     attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
 to_0                    ->  +     ...
 attn_weights_2          ->  +     ...
 nn_functional_dropout_0 -> 17     attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
 attn_weights_3          ->  +     ...
 torch_matmul_1          -> 18     attn_output = torch.matmul(attn_weights, value_states)
 attn_output_0           ->  +     ...
 attn_output_transpose_0 -> 19     attn_output = attn_output.transpose(1, 2).contiguous()
 contiguous_0            ->  +     ...
 attn_output_1           ->  +     ...
                            21     return attn_output, attn_weights
```

This listing is where the base `Attention`'s values come from: `attention_scores` is
`nn_functional_softmax_0`'s input (the masked `attn_weights_1`), `attention_probabilities`
is `nn_functional_dropout_0`'s output, `attention_head_outputs` is the call's own
`output` element 0. GPT-OSS's interface concatenates a sink column between
`attn_weights_1` and the softmax, so its family reads `attention_scores` at
`attn_weights_1` by name ([overriding-values.md](overriding-values.md)).

An `EProperty` does this drill for you at every read, with a `source` segment in the path
separating the call from the operation inside it:
`"source.attention_interface_1.source.nn_functional_dropout_0.output"`.

## How operations are named

- **A call** is `<callee>_<n>`, the whole attribute chain joined with `_`: `self.c_proj(...)`
  is `self_c_proj_0`, `nn.functional.softmax(...)` is `nn_functional_softmax_0`,
  `self._attn(...)` is `self__attn_0` (the underscore in `_attn` stays), `torch.bmm(...)` is
  `torch_bmm_0`, `F.softmax(...)` is `F_softmax_0` when the module imports it as `F`. The
  counter is per callee, in the order the calls appear in the forward's source, not the
  order they run: Falcon's attention has one `F.softmax` per branch, and on an alibi
  checkpoint the one that runs is `F_softmax_1`, the second in the source.
- **A binding** is an operation too, `<name>_<n>`: the n-th assignment to that name in the
  forward. `query_states_0` is the first binding of `query_states`; `attn_weights_1` is the
  second binding of `attn_weights` (after the mask). Calls and bindings share one counter
  per name, so `attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(...)` is
  `attention_interface_0` and the call `attention_interface(...)` is `attention_interface_1`.
  The `+ ...` lines under a call are the bindings of its result.
- **`.output`** is what the call returned (or what the binding bound); **`.input`** the
  call's first argument; **`.inputs`** the full `(args, kwargs)`. For a binding, `.input`
  and `.output` are the same value.

Families use both kinds: MPT reads its queries at the `query_states_0` binding, Falcon
its values at `value_layer_0` and its head outputs at the `attn_output_1` binding;
BLOOM reads its `dropout_add_0` call's input.

## Picking the operation

- **The pattern is the dropout after the softmax**, where one exists: `nn_functional_dropout_0`
  on the interface, `self_attention_dropout_0` on BLOOM, `self__attn_0.source.self_attn_dropout_0`
  on GPT-J. That is the tensor the values are mixed with, in the model's dtype (the softmax
  runs in float32 on the interface and casts back) and, on a sink model, with the sink
  column already dropped. In eval on a float32 model the softmax output and the dropout
  output are equal; the rule is about what the location means, not the numbers. Falcon
  without alibi has no dropout after `F_softmax_0`, so its pattern is the softmax itself;
  with alibi its pattern is `self_attention_dropout_0`, after the second softmax.
- **A contribution added inside the module** is the tensor entering the add: BLOOM's
  `dropout_add_0` input, MPT's `F_dropout_0` output.
- **Queries, keys and values** are what the attention arithmetic receives: the interface
  call's arguments, GPT-J's `self__attn_0` arguments, BLOOM's `self__reshape_0` returns,
  Falcon's `apply_rotary_pos_emb_0` returns (its `query_layer_0` / `key_layer_0` bindings
  with alibi, where no rotary runs). Take them after the rotary embedding where there is
  one, before `repeat_kv` so the head axis is `num_kv_heads` wide.
- **Head outputs** are the last matmul: the interface call's output element 0,
  `torch_bmm_0` on BLOOM, `torch_matmul_1` on MPT.

## What a wrong name tells you

- A name not in the listing raises `AttributeError` naming every operation the module has:
  `'model.transformer.h.0.attn.source' has no operation 'nope_0'; available: is_cross_attention_0, ...`.
- An `EProperty` whose op is missing raises `SourceNotAvailable` with the value's
  path, its key, and that list, plus "The forward took a path this family's toolkit does
  not expect."
- A name that is in the listing but under a branch this forward does not take
  (`self__upcast_and_reordered_attn_0` when `reorder_and_upcast_attn` is off, GPT-2's
  cross-attention ops `self_q_attn_0`, `self_c_attn_0`, `transpose_0`) fails with
  `OutOfOrderError`: the model ran past a location it never reached. The live twin is
  the next occurrence of the same callee (`self_c_attn_1`, `transpose_2`).
- `nn_functional_softmax_0.source` cannot be drilled: `torch.nn.functional` entry points
  refer to their own name to pass themselves to the dispatcher. Their `.output` and
  `.input` are what a value wants anyway.
- Under `attn_implementation="sdpa"` the interface is a fused kernel with no pattern to
  read. `needs_eager` is the reason the base values report; a value of your own on the
  interface takes `unavailable=interface_reason`.

## Order within a trace

Requests are served in the forward's order. For a call and its inside: the call's
`.inputs`, then the drill (`.source`, served from the call's `.fn` just before it runs),
then the operations inside, then the call's `.output`. The Llama snippet above reads them
in that order; reading `attention_interface_1.inputs` after an operation inside the call
raises `OutOfOrderError` on `...attention_interface_1.input.i0`. An `EProperty` drills
at every read, so the same rule governs two values read in one trace: `attention_queries`
(the call's inputs) before `attention_probabilities` (inside the call), and on Falcon
`attention_values` (the `value_layer_0` binding) before the rotary's returns. The family
suite reads interior values one per trace for this reason.

## Names move with transformers

The operation names are what transformers releases change: a renamed local variable
renames a binding, a refactor that routes a family onto the shared interface (DBRX on
5.17) replaces its own softmax with `attention_interface_1`, a moved dropout changes the
pattern's op. Print the source for the version you run rather than copying a label from
another family or from this page, and let `tests/families/test_<family>.py`'s
`test_every_source_value_resolves_on_every_layer` and
`test_written_pattern_moves_the_logits` be the guard: every source-located value must
resolve on every attention block, and a written pattern must move the logits, since an
address can read a perfectly good tensor nothing downstream uses.

## Gotchas

- **Print outside a trace, drill inside one.** `print(envoy.source)` needs no trace;
  `call.source` needs the live callee.
- **The listing includes dead branches.** GPT-2's attention lists its cross-attention and
  cache-hit paths beside the live ones; a request on a dead label is an `OutOfOrderError`,
  not an `AttributeError`.
- **A submodule call is not drilled.** `self_c_proj_0.source` is refused; read
  `attn.c_proj.output` or `.source` on that submodule.
- **First `.source` access on a module must come before that module's forward runs** in a
  trace, since it rewrites the forward. A value on the module's own `source.` path read as
  the first request on its module is fine; a bare `_ = envoy.source` outside the trace
  instruments it up front, and `sourced = True` on a `Standard` subclass does that at
  build, for a value in its forward that is read after the call has begun (Llama 4's
  `Layer`, whose `mlp_output` follows `attention_output`).
- **Drill into a call on step 0 if you will read inside it with `tracer.iter`.** Occurrences
  of an operation inside a called function are counted from the first `op.source` drill of
  the run, not from the run's start. Under `generate`, a call first drilled on step 1 has
  that step as its occurrence 0, so a later `tracer.iter[1]` read inside it returns step
  2's value, without an error. Touch `op.source` before the first step, or read it on
  every step from 0.
- **Sourcing a module costs a little on every forward afterwards**, trace or not
  (nnsight docs/usage/source.md quotes about 6% for all of GPT-2's blocks); a family
  value instruments only the modules it is read on.

## Related

- [overriding-values.md](overriding-values.md): the descriptors that take these names.
- [custom-values.md](custom-values.md): an `EProperty` of your own on an op you found.
- [adding-a-family.md](adding-a-family.md): the suite test that guards every op name.
- nnsight docs/usage/source.md: operation naming, iteration, dispatchers and the full gotcha list.
