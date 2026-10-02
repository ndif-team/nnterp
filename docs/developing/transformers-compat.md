---
title: transformers Compatibility
one_liner: The versions nnter is developed against, exactly which operation names a transformers release can move, how the suite catches it, and the upgrade procedure.
tags: [developing, compatibility, transformers, nnsight, versions, source]
related: [docs/developing/testing.md, docs/developing/eproperty-internals.md, docs/developing/architecture.md, docs/developing/gotchas.md]
sources: [nnter/components/attention.py, nnter/components/linear_attention.py, nnter/components/recurrent.py, nnter/families/gpt_oss.py, nnter/families/bloom.py, nnter/families/mpt.py, nnter/families/falcon.py, nnter/families/gptj.py, tests/families/suite.py, tests/conftest.py, pyproject.toml]
---

# transformers Compatibility

## What this is for

This is the one page in `docs/` that talks about versions. nnter is
developed on **transformers 5.17.0** and **nnsight 0.8.0** (the `dev`
branch, at the eproperty-transform-raw merge), with torch 2.13 and
jaxtyping 0.3.11; `pyproject.toml` requires `nnsight>=0.8`, `transformers`
and `jaxtyping`. The names a family binds to (`RENAME`, `ENVOYS`) are
transformers' module names and classes, which move rarely. The **operation
names** inside a forward that every `source.` path pins are what a
release renames, and this page lists them, says how the suite guards them,
and gives the procedure when one moves.

## Canonical pattern

What the family pins, next to what the live forward has (run on
`hf-internal-testing/tiny-random-LlamaForCausalLM`; a Llama-family model):

```python
import nnter
from nnter import StandardizedTransformer
from nnter.components import INTERFACE

model = StandardizedTransformer("meta-llama/Llama-3.1-8B", dispatch=True, attn_implementation="eager")
attn = model.layers[0].self_attn

[op.name for op in attn.source]                     # outside a trace: the module's own forward
# [..., 'apply_rotary_pos_emb_0', 'past_key_values_update_0', 'ALL_ATTENTION_FUNCTIONS_get_interface_0',
#  'attention_interface_0', 'attention_interface_1', 'attn_output_reshape_0', ...]

with model.trace("Hello world there"):
    inner = [op.name for op in attn.source.attention_interface_1.source].save()   # inside: drilled from the live callee
# ['repeat_kv_0', 'key_states_0', 'repeat_kv_1', 'value_states_0', 'key_states_transpose_0', 'torch_matmul_0',
#  'attn_weights_0', 'attn_weights_1', 'nn_functional_softmax_0', 'to_0', 'attn_weights_2',
#  'nn_functional_dropout_0', 'attn_weights_3', 'torch_matmul_1', 'attn_output_0', ...]

{name: attr.key for name, attr in type(attn).values().items() if attr.inside_forward()}
# {'attention_queries':       'source.attention_interface_1.inputs',
#  'attention_keys':          'source.attention_interface_1.inputs',
#  'attention_values':        'source.attention_interface_1.inputs',
#  'attention_scores':        'source.attention_interface_1.source.nn_functional_softmax_0.input',
#  'attention_probabilities': 'source.attention_interface_1.source.nn_functional_dropout_0.output',
#  'attention_head_outputs':  'source.attention_interface_1.output'}
```

`print(attn.source)` renders the forward with every op labelled at its
line (nnsight `source.py:1040`); that is the view to diff across releases.

## What a release can break

An op name is `{callable}_{occurrence}`, where the callable is the dotted
name in the source joined with `_` and the occurrence counts calls **and
bindings of that name** in execution order (nnsight `source.py:15-31`).
Three things therefore move a name: the call is renamed, a binding of the
same name is added or removed before it, or the arithmetic moves onto or
off transformers' shared attention interface.

### The shared interface (most families)

`INTERFACE = "attention_interface_1"` (`nnter/components/attention.py:26`).
The module binds `attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(...)`
(`attention_interface_0`) and then calls it (`attention_interface_1`);
`modeling_llama.py:264-268` in 5.17. Inside `eager_attention_forward`
(`modeling_llama.py:191-209`) the pattern is `nn.functional.softmax(...)`
then `nn.functional.dropout(...)`: `nn_functional_softmax_0` (its `.input`
is `attention_scores`) and `nn_functional_dropout_0` (its `.output` is
`attention_probabilities`, `attention.py:154`, `:179-181`). The interface's
`inputs` positions 1, 2, 3 are queries, keys, values, and `output[0]` is the
head outputs (`:74-103`, `:149`).

**GPT-OSS** keeps the interface but its eager forward concatenates a sink
column before the softmax (`modeling_gpt_oss.py:247-260`); the standard
`attention_scores` is the masked scores bound just before that,
`attn_weights_1` (`nnter/families/gpt_oss.py:43-45`), which exists only when
an attention mask is passed, as it is on every prompt.

### Families with their own attention arithmetic

| family | value → op (`nnter/families/<f>.py`) |
|---|---|
| **BLOOM** (`bloom.py:44-84`, `:90-96`) | q/k/v `self__reshape_0.output[0..2]`; scores `F_softmax_0.input`; head outputs `torch_bmm_0.output`; pattern `self_attention_dropout_0.output`; both contributions `dropout_add_0.input` |
| **MPT** (`mpt.py:45-74`, `:80-85`) | q/k/v the bindings `query_states_0`, `key_states_0`, `value_states_0`; scores `nn_functional_softmax_0.input`; head outputs `torch_matmul_1`; pattern `nn_functional_dropout_0`; MLP contribution `F_dropout_0` |
| **Falcon** (`falcon.py:42-119`) | each op chosen by `config.alibi` through `by_alibi(without, with_alibi)`. Without alibi: queries/keys `apply_rotary_pos_emb_0.output[0/1]`; values the binding `value_layer_0`; scores `F_softmax_0.input`; head outputs `attn_output_1` (the `attention_scores @ value_layer` binding; `attn_output_0` is the sdpa branch that does not run, `modeling_falcon.py:350`, `:365`); pattern `F_softmax_0.output` (no dropout follows). With alibi: queries/keys/values the bindings `query_layer_0`, `key_layer_0`, `value_layer_0`; scores `F_softmax_1.input`; pattern `self_attention_dropout_0.output`; head outputs `flatten_0.output` (`[batch * heads, seq, head_dim]`, served as a `[batch, seq, heads, head_dim]` view) |
| **GPT-J** (`gptj.py:43-73`) | everything around the `self._attn(...)` call: `self__attn_0.inputs[0..2]`, `self__attn_0.source.nn_functional_softmax_0.input`, `self__attn_0.output[0]`, `self__attn_0.source.self_attn_dropout_0` |
| **GPT-Neo** (`gpt_neo.py`) | the same ops as GPT-J on the inner `GPTNeoSelfAttention` (`attn.attention`), which the `GPTNeoAttention` wrapper calls |
| **CodeGen** (`codegen.py`) | GPT-J's ops around `self__attn_0`, except the scores: `self__attn_0.source.call_0.input`, the call of the `nn.Softmax(dim=-1)` instance built on the same line (`nn_Softmax_0` is the construction) |
| **GPT-NeoX-Japanese** (`gpt_neox_japanese.py`) | around `self__attn_0`: `inputs[0..2]`, `source.nn_functional_softmax_0.input`, `output[0]`, `source.self_attention_dropout_0` |
| **XGLM** (`xglm.py`) | q/k/v `query_states_reshape_0` / `key_states_reshape_0` / `value_states_reshape_0` outputs; scores `nn_functional_softmax_1.input` (`_0` is the fp16 upcast branch, chosen by the weights' dtype); pattern `nn_functional_dropout_0.output`; head outputs `torch_bmm_1.output`; every one viewed from `[batch * heads, ...]` |
| **DBRX** (`dbrx.py:10`) | on the shared interface in 5.17, so the base class holds; an earlier layout had its own arithmetic |

### Gated DeltaNet (Qwen3-Next, Qwen3.5, Qwen3.5-MoE)

`torch_chunk_gated_delta_rule_0` and `torch_recurrent_gated_delta_rule_0`
are the two kernel calls, `use_precomputed_states_0` the branch binding that
picks between them, and `last_recurrent_state_3` the per-token state
binding inside the recurrent loop
(`nnter/components/linear_attention.py:45-49`;
`modeling_qwen3_5.py:561`, `:626`, `:639`, `:474-494`). The state op's `_3`
is the count of `last_recurrent_state = ...` bindings before the one after
the token's update: two before the loop, the decay inside it, then the
update. A release that adds or removes one binding moves it. The kernels'
argument names `g`, `beta`, `initial_state` are the `select` keys of
`decays`, `betas`, `state_input` (`linear_attention.py:66-85`). The
kernel-dispatch closure names `torch_function` and `implementation`
(`transformers/integrations/hub_kernels.py:847-859`) are what
`needs_torch_kernels`, `route_kernels` and `needs_recurrent_routing` read
through `_dispatch` (`recurrent.py:46-58`, `:95-187`).

### Names, not ops

`RENAME` keys and `ENVOYS` classes are module names and types; a rename
there is a load-time failure (an alias that resolves nowhere is silently
skipped, so `test_standard_names_alias_native_envoys` is what catches it).
GPT-NeoX's head is `lm_head` now and was `embed_out`; both spellings are
keys, and nnsight binds whichever resolves (`gpt_neox.py:17-20`).

## How the suite guards it

`tests/families/suite.py` runs on every family's pinned checkpoint
([testing.md](testing.md)):

- `test_every_source_value_resolves_on_every_layer` (`suite.py:320-335`):
  every available value on the family's `Attention` whose path is inside a
  forward reads a tensor on every attention block. A renamed op fails here
  with `SourceNotAvailable` naming the missing op and the ops that exist
  (`nnter/components/eproperty.py:186-190`).
- `test_written_pattern_moves_the_logits` (`:303-317`): assigning a random
  pattern and zeroing a head in place both move the logits. An op that
  still resolves but is no longer what the values are mixed with (a copy, a
  tensor the forward returns and never uses) reads fine and fails here.
- `test_interior_shapes` (`:348-359`) checks `softmax(scores) == pattern`,
  which catches a `_0`/`_1` slip that lands on a different tensor of the
  same shape.
- `test_contribution_identity` (`:208-218`) catches a moved contribution op
  on BLOOM, MPT and Falcon.
- The DeltaNet tests (`tests/families/test_qwen3_5_text.py:44-63`,
  `:93-109`, `:118-148`) catch the kernel names, the branch binding, the
  argument names and the state binding.

## Procedure when upgrading transformers

1. Upgrade in a scratch environment and run `HF_HUB_OFFLINE=1 pytest -q`.
   Read the `SourceNotAvailable` messages: each names the value, the op it
   pinned, and the ops the forward has now.
2. For each failing family, `print(model.layers[0].self_attn.source)` on
   the old and new versions (and, inside a trace,
   `print(attn.source.attention_interface_1.source)` for the interface) and
   diff the labelled listings. Decide whether the value moved (a renamed
   call, a binding added before it) or the family moved onto or off the
   shared interface (DBRX's history).
3. Update the op string in the family module, or delete the override when
   the family now uses the shared interface. Keep the standard name's
   meaning fixed: the pattern is the tensor the values are mixed with, the
   scores are the softmax's input, the contribution is what is added to the
   stream (`components/attention.py:64-67`).
4. Re-run the family's file, then the whole suite. Update the version
   sentence at the top of this page.
5. A checkpoint whose config no longer parses is a test-data problem, not a
   family problem: patch the config the way `tests/families/test_olmo3.py:14-32`
   does.

## The import-order segfault

On this stack, importing a `transformers.models.*.modeling_*` module
**before** nnsight segfaults at import. `tests/conftest.py` imports nnsight
first in every pytest process; every family module is imported only through
`nnter.families.lookup`, after `import nnter`; a script must `import nnter`
(or `import nnsight`) before any `from transformers.models... import`. A
plain `import transformers` first is fine.

## What nnter needs from nnsight

Two behaviours of nnsight 0.8 as it stands, both used by nnter as current
behaviour:

- **`envoys=` chooses the envoy class per module at construction**, by type
  (MRO) first and native path suffix second, before aliases bind
  (nnsight `envoy.py:224-230`, `:349-370`, `:177`). nnter keys every
  `ENVOYS` on a transformers type; an alias never matches; a user displaces a
  family envoy by keying on the same type (`tests/test_registry.py:67-77`).
- **`eproperty.transform` takes `(self, view, raw)`**: the edited view and
  the value as served (nnsight `eproperty.py:158-166`, `:178-184`). Falcon's
  `mlp_output` uses it to carry an in-place edit on a clone back into the
  model (`nnter/families/falcon.py:136-148`).

Also relied on: `Mediator.current`, `Mediator.iteration`, `Mediator.occurrence`,
`Interleaver.sourced` and `Iterations` (nnsight `interleaver.py:287-334`,
`:654-660`; `iterator.py:103-144`), which are not part of nnsight's
documented public surface; [eproperty-internals.md](eproperty-internals.md)
and [recurrent-mixer-internals.md](recurrent-mixer-internals.md) say
exactly how.

## Gotchas

- The occurrence suffix counts bindings too: `attention_interface_1` is the
  call because `attention_interface_0` is the binding.
- `attn_weights_1` on GPT-OSS is the *masked* scores; with no attention mask
  the binding at `modeling_gpt_oss.py:249` does not run and the name shifts.
- Falcon's `attn_output_0` is a branch that does not execute under eager;
  nnsight numbers every call in the source, executed or not.
- A `SourceNotAvailable` is only raised inside a trace; `support()` cannot
  know an op moved, since it does not run the forward. The suite is the
  guard.
- Do not add version conditionals to family modules; one family module
  targets the transformers nnter is developed on, and the upgrade procedure
  moves it.

## Related

- [testing.md](testing.md) — the guard, method by method
- [eproperty-internals.md](eproperty-internals.md) — `_resolve`'s walk and `SourceNotAvailable`
- [recurrent-mixer-internals.md](recurrent-mixer-internals.md) — the DeltaNet names in context
- nnsight `docs/usage/source.md`, `docs/developing/source-internals.md` — how op labels are made
