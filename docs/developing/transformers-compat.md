---
title: transformers Compatibility
one_liner: The versions nnterp is developed against, exactly which operation names a transformers release can move, how the suite catches it, and the upgrade procedure.
tags: [developing, compatibility, transformers, nnsight, versions, source]
related: [docs/developing/testing.md, docs/developing/eproperty-internals.md, docs/developing/architecture.md, docs/developing/gotchas.md]
sources: [nnterp/components/attention.py, nnterp/components/linear_attention.py, nnterp/components/recurrent.py, nnterp/families/gpt_oss.py, nnterp/families/bloom.py, nnterp/families/mpt.py, nnterp/families/falcon.py, nnterp/families/gptj.py, tests/families/suite.py, tests/conftest.py, pyproject.toml]
---

# transformers Compatibility

## What this is for

This is the one page in `docs/` that talks about versions. nnterp is
developed on **transformers 5.17.0** and **nnsight 0.8.0**, with torch 2.13 and
jaxtyping 0.3.11, and runs on **transformers 5.19.0** and **5.20.1** as well: the suite passes on
all three (a test that needs a 5.18 class skips on 5.17; on 5.20 the SmolVLM tower tests skip, because 5.20's
`inputs_merger` reshapes the image features by `vision_config.patch_size` rather than by the features each image
yields, which fails when the two differ, and Kimi K2.5's `text_config.model_type` became `deepseek_v3`, so
the wrapper resolves through a family module of its own, `kimi_k25.py`, which re-exports `kimi_k2`'s; `nnterp.families.resolve`
tries a config's own type before its text model's), and every snippet in `docs/` that runs on CPU with a tiny checkpoint returns
identical values on both wherever they do not depend on unseeded randomness. On **5.21.0** the recurrent mixers hold
too: it dropped the `use_precomputed_states` binding from every recurrent forward for a fetch from the cache, which
`RecurrentMixer.BRANCH` now names beside the old binding ([below](#gated-deltanet-qwen3-next-qwen35-qwen35-moe)), and
the recurrent families' tests pass there; Jamba's router became a ``JambaTopKRouter`` module there (a bare ``nn.Linear``
before), so its `router_logits` read whichever the mixture has; SmolVLM's model scatters its image features itself there
(``masked_scatter``, as Llava does) where 5.20 called ``inputs_merger``, so its scatter host reads whichever operation
the forward has, and the 5.20-only SmolVLM reshape bug that skipped its tower tests is fixed in 5.21. The suite passes
on 5.21.0 with 0 failures. `pyproject.toml` requires `nnsight>=0.8`,
`transformers>=5.17` and `jaxtyping`. nnterp needs an nnsight whose `Envoy` has `parent` and `root`
(below) and whose `rename` takes a dotted alias (`{"attn.c_proj":
"attn.o_proj"}` binds on the attention, which GPT-2's two `c_proj` need); nnsight 0.8.0
has both, which is what `nnsight>=0.8` says. The names a family binds to (`RENAME`, `ENVOYS`) are
transformers' module names and classes, which move rarely. The **operation
names** inside a forward that every `source.` path pins are what a
release renames, and this page lists them, says how the suite guards them,
and gives the procedure when one moves.

## Canonical pattern

What the family pins, next to what the live forward has (run on
`hf-internal-testing/tiny-random-LlamaForCausalLM`; a Llama-family model):

```python
import nnterp
from nnterp import StandardizedTransformer
from nnterp.components import INTERFACE

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
# {'attention_queries_prerope': 'source.apply_rotary_pos_emb_0.inputs',
#  'attention_keys_prerope':  'source.apply_rotary_pos_emb_0.inputs',
#  'attention_queries':       'source.attention_interface_1.inputs',
#  'attention_keys':          'source.attention_interface_1.inputs',
#  'attention_values':        'source.attention_interface_1.inputs',
#  'attention_scores':        'source.attention_interface_1.source.nn_functional_softmax_0.input',
#  'attention_probabilities': 'source.attention_interface_1.source.nn_functional_dropout_0.output',
#  'attention_head_outputs':  'source.attention_interface_1.output'}
```

`print(attn.source)` renders the forward with every op labelled at its
line (`Source.__repr__`, nnsight `source.py`); that is the view to diff
across releases.

## What a release can break

An op name is `{callable}_{occurrence}`, where the callable is the dotted
name in the source joined with `_` and the occurrence counts calls **and
bindings of that name** in execution order (`Instrument`, nnsight `source.py`).
Three things therefore move a name: the call is renamed, a binding of the
same name is added or removed before it, or the arithmetic moves onto or
off transformers' shared attention interface.

### The shared interface (most families)

`INTERFACE = "attention_interface_1"` (`nnterp/components/attention.py`).
The module binds `attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(...)`
(`attention_interface_0`) and then calls it (`attention_interface_1`);
`LlamaAttention.forward`, `modeling_llama.py` in 5.17. Inside
`eager_attention_forward` (`modeling_llama.py`) the pattern is
`nn.functional.softmax(...)` then `nn.functional.dropout(...)`:
`nn_functional_softmax_0` (its `.input` is `attention_scores`) and
`nn_functional_dropout_0` (its `.output` is `attention_probabilities`,
`attention.py`). The interface's `inputs` positions 1, 2, 3 are queries,
keys, values, and `output[0]` is the head outputs
(`Attention.attention_queries`, `attention_keys`, `attention_values`,
`attention_head_outputs`).

**GPT-OSS** keeps the interface but its eager forward concatenates a sink
column before the softmax (`eager_attention_forward`, `modeling_gpt_oss.py`);
the standard `attention_scores` is the masked scores bound just before that,
`attn_weights_1` (`nnterp/families/gpt_oss.py`), which runs only when
an attention mask is passed, as it is on every prompt.

### Families with their own attention arithmetic

| family | value → op (`nnterp/families/<f>.py`) |
|---|---|
| **BLOOM** (`Attention`, `Mlp`, `bloom.py`) | q/k/v `self__reshape_0.output[0..2]`; scores `F_softmax_0.input`; head outputs `self__merge_heads_0.output` viewed per head; pattern `self_attention_dropout_0.output`; both contributions `dropout_add_0.input` |
| **MPT** (`Attention`, `Mlp`, `mpt.py`) | q/k/v the bindings `query_states_0`, `key_states_0`, `value_states_0`; scores `nn_functional_softmax_0.input`; head outputs `torch_matmul_1`; pattern `nn_functional_dropout_0`; MLP contribution `F_dropout_0` |
| **Falcon** (`Attention`, `falcon.py`) | each op chosen by `config.alibi` through `by_alibi(without, with_alibi)`. Without alibi: queries/keys `apply_rotary_pos_emb_0.output[0/1]`; values the binding `value_layer_0`; scores `F_softmax_0.input`; head outputs `attn_output_1` (the `attention_scores @ value_layer` binding; `attn_output_0` is the sdpa branch that does not run, `FalconAttention.forward`, `modeling_falcon.py`); pattern `F_softmax_0.output` (no dropout follows). With alibi: queries/keys/values the bindings `query_layer_0`, `key_layer_0`, `value_layer_0`; scores `F_softmax_1.input`; pattern `self_attention_dropout_0.output`; head outputs `self__merge_heads_0.output` (`[batch, seq, num_heads * head_dim]`, served as a `[batch, seq, num_heads, head_dim]` view; the product before it, `flatten_0`, is `[batch * num_heads, ...]`) |
| **GPT-J** (`Attention`, `gptj.py`) | everything around the `self._attn(...)` call: `self__attn_0.inputs[0..2]`, `self__attn_0.source.nn_functional_softmax_0.input`, `self__attn_0.output[0]`, `self__attn_0.source.self_attn_dropout_0` |
| **GPT-Neo** (`gpt_neo.py`) | the same ops as GPT-J on the inner `GPTNeoSelfAttention` (`attn.attention`), which the `GPTNeoAttention` wrapper calls |
| **CodeGen** (`codegen.py`) | GPT-J's ops around `self__attn_0`, except the scores: `self__attn_0.source.call_0.input`, the call of the `nn.Softmax(dim=-1)` instance built on the same line (`nn_Softmax_0` is the construction) |
| **GPT-NeoX-Japanese** (`gpt_neox_japanese.py`) | around `self__attn_0`: `inputs[0..2]`, `source.nn_functional_softmax_0.input`, `output[0]`, `source.self_attention_dropout_0` |
| **XGLM** (`xglm.py`) | every value but the keys and values where the batch still leads (the `bmm` calls take `[batch * num_heads, ...]`): queries the `query_states_0` binding, viewed heads first; keys/values `key_states_reshape_0` / `value_states_reshape_0` outputs, after the cache update, viewed from `[batch * num_heads, ...]`; scores the `attn_weights_2` binding (the clamp, before either softmax branch); pattern the `attn_weights_reshaped_0` binding (`nn_functional_dropout_0.output`, viewed, in training); head outputs the `attn_output_2` binding |
| **DBRX** (`Attention`, `dbrx.py`) | on the shared interface in 5.17, so the base class holds; an earlier layout had its own arithmetic |

### Gated DeltaNet (Qwen3-Next, Qwen3.5, Qwen3.5-MoE)

`torch_chunk_gated_delta_rule_0` and `torch_recurrent_gated_delta_rule_0`
are the two kernel calls, the cached-state op picks between them
(`use_precomputed_states_0` through 5.20; 5.21 dropped that binding from
every recurrent forward for a fetch from the cache,
`cache_params_get_recurrent_state_0`, made only under `if cache_params is
not None:`, so `RecurrentMixer.BRANCH` names both and `KERNEL` reads
whichever the forward has), and `last_recurrent_state_3` the per-token state
binding inside the recurrent loop
(`LinearAttention`, `nnterp/components/linear_attention.py`;
`Qwen3_5GatedDeltaNet.forward`, `torch_recurrent_gated_delta_rule`,
`modeling_qwen3_5.py`). The state op's `_3`
is the count of `last_recurrent_state = ...` bindings before the one after
the token's update: two before the loop, the decay inside it, then the
update. A release that adds or removes one binding moves it. The kernels'
argument names `g`, `beta`, `initial_state` are the `select` keys of
`decays`, `betas`, `state_input` (`linear_attention.py`). The
kernel-dispatch closure names `torch_function` and `implementation`
(`use_kernel_func_from_hub_with_fallback`,
`transformers/integrations/hub_kernels.py`) are what
`needs_torch_kernels`, `route_kernels` and `needs_recurrent_routing` read
through `_dispatch` (`recurrent.py`).

### Names, not ops

`RENAME` keys and `ENVOYS` classes are module names and types; a rename
there is a load-time failure (an alias that resolves nowhere is silently
skipped, so `test_standard_names_alias_native_envoys` is what catches it).
GPT-NeoX's head is `lm_head` now and was `embed_out`; the old spelling is
a key, bound where it resolves (`RENAME`, `gpt_neox.py`).

## How the suite guards it

`tests/families/suite.py` runs on every family's pinned checkpoint
([testing.md](testing.md)):

- `test_every_source_value_resolves_on_every_layer` (`suite.py`):
  every available value on the family's `Attention` whose path is inside a
  forward reads a tensor on every attention block. A renamed op fails here
  with `SourceNotAvailable` naming the missing op and the ops that exist
  (`EProperty._resolve`, `nnterp/components/eproperty.py`).
- `test_written_pattern_moves_the_logits`: assigning a random
  pattern and zeroing a head in place both move the logits. An op that
  still resolves but is no longer what the values are mixed with (a copy, a
  tensor the forward returns and never uses) reads fine and fails here.
- `test_interior_shapes` checks `softmax(scores) == pattern`,
  which catches a `_0`/`_1` slip that lands on a different tensor of the
  same shape.
- `test_contribution_identity` catches a moved contribution op
  on BLOOM, MPT and Falcon.
- The DeltaNet tests (`TestQwen3_5.test_linear_values_shapes`,
  `TestQwen3_5.test_values_follow_the_step_under_generate`,
  `TestQwen3_5.test_per_token_state_with_the_recurrent_kernel`,
  `tests/families/test_qwen3_5_text.py`) catch the kernel names, the branch
  binding, the argument names and the state binding.

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
   stream (`Attention`, `components/attention.py`).
4. Re-run the family's file, then the whole suite. Update the version
   sentence at the top of this page.
   Releases also move behaviour no op name pins. On 5.18 and later `generate`
   calls the model with no `attention_mask` when the batch has no padding
   (`model.attention_mask` serves the all-ones mask it would have had,
   [root-values.md](../usage/root-values.md)), and encodes a wrapper's images before
   the first forward, so under `generate` the tower's values come before
   `vision.image_token_mask` ([vision.md](../usage/vision.md)).
5. A checkpoint whose config no longer parses is a test-data problem, not a
   family problem: patch the config the way `_patched_checkpoint`
   (`tests/families/test_olmo3.py`) does.

## What nnterp needs from nnsight

Three behaviours of nnsight 0.8 as it stands, all used by nnterp as current
behaviour:

- **`Envoy.parent` and `Envoy.root`**: every envoy keeps a weak link to the
  envoy that holds it, and `root` follows the links up to the model envoy.
  A value reads its block through `self.parent` (the Granite
  mixtures' `residual_multiplier`, the Gemma-4 and GraniteMoE-Hybrid `Mlp`'s
  `router` and `experts`, ZAYA's `merge`) and the model through `self.root`
  (`Vision`'s config, processor and family), and a key with a leading `/` is
  walked from `root` (`"/inputs"`). The link is dropped when an
  envoy is pickled and rebuilt inside the payload, so a shipped tree whose top
  is the model envoy keeps its root; a remote trace ships the model's whole
  tree, so a deserialized tower's root is the deserialized
  `StandardizedTransformer`.

- **`envoys=` chooses the envoy class per module at construction**, by type
  (MRO) first and path suffix second, the native path or any alias spelling
  of it (`Envoy._wrap_envoy`, `Envoy._resolve_envoy_class`, `Envoy.__init__`,
  nnsight `envoy.py`). nnterp keys every `ENVOYS` on a transformers type, so
  no path key displaces a family envoy; a user displaces one by keying on
  the same type (`test_user_envoys_merge_over_defaults`,
  `tests/test_registry.py`).
- **`eproperty.transform` takes `(self, view, raw)`**: the edited view and
  the value as served (`eproperty.__get__`, nnsight `eproperty.py`). Falcon's
  `mlp_output` uses it to carry an in-place edit on a clone back into the
  model (`nnterp/families/falcon.py`).

Also relied on: `Mediator.current`, `Mediator.iteration`, `Mediator.occurrence`,
`Interleaver.sourced` and `Iterations` (nnsight `interleaver.py`;
`iterator.py`), which are not part of nnsight's
documented public surface; [eproperty-internals.md](eproperty-internals.md)
and [recurrent-mixer-internals.md](recurrent-mixer-internals.md) say
exactly how.

## Gotchas

- The occurrence suffix counts bindings too: `attention_interface_1` is the
  call because `attention_interface_0` is the binding.
- `attn_weights_1` on GPT-OSS is the *masked* scores; with no attention mask
  the binding in `eager_attention_forward` (`modeling_gpt_oss.py`) does not
  run.
- Falcon's `attn_output_0` is a branch that does not execute under eager;
  nnsight numbers every call in the source, executed or not.
- A `SourceNotAvailable` is only raised inside a trace; `support()` cannot
  know an op moved, since it does not run the forward. The suite is the
  guard.
- Do not add version conditionals to family modules; one family module
  targets the transformers nnterp is developed on, and the upgrade procedure
  moves it.

## Related

- [testing.md](testing.md) — the guard, method by method
- [eproperty-internals.md](eproperty-internals.md) — `_resolve`'s walk and `SourceNotAvailable`
- [recurrent-mixer-internals.md](recurrent-mixer-internals.md) — the DeltaNet names in context
- nnsight `docs/usage/source.md`, `docs/developing/source-internals.md` — how op labels are made
