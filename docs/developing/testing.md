---
title: Testing
one_liner: Running nnterp's suite offline, what FamilySuite asserts method by method, how a family file adds its specifics, and the root tests for the registry, descriptors and helpers.
tags: [developing, testing, pytest, families]
related: [docs/developing/contributing.md, docs/developing/architecture.md, docs/developing/transformers-compat.md, docs/extending/index.md]
sources: [tests/conftest.py, tests/families/suite.py, tests/families/test_falcon.py, tests/families/test_olmo3.py, tests/families/test_gpt2.py, tests/test_registry.py, tests/test_base.py, tests/test_prompt_utils.py, tests/test_nnsight_utils.py, pyproject.toml]
---

# Testing

## What this is for

The suite is the executable contract of nnterp: every statement the README
makes about a value is a method in `tests/families/suite.py`, run against a
pinned tiny checkpoint of every family. Adding a family is one test file;
adding a value is one method here. This page is how to run it, what each
test asserts, and how a family file states its specifics.

## Canonical pattern

```bash
cd nnterp                                # the repository root
export HF_HUB_OFFLINE=1
pytest                                   # 5221 passed, 906 skipped in ~395 s (6 min 40 s wall) on CPU
pytest tests/families/test_gpt2.py -q    # one family: 37 passed, 12 skipped in 3.0 s (7.1 s wall)
pytest tests/test_registry.py -q         # 14 passed in 10.6 s (one test spawns a subprocess)
pytest "tests/families/test_gpt2.py::TestGPT2::test_contribution_identity" -q -x   # one test, stop on first failure
```

Timings are from `/home/localjadenfk/miniconda3/envs/ndif2` (the stack
[transformers-compat.md](transformers-compat.md) names) on this machine's
CPU; the wall time above the reported one is interpreter and import start-up. Every
checkpoint is cached, so `HF_HUB_OFFLINE=1` is what keeps a run from
touching the Hub. `pyproject.toml` configures only `testpaths = ["tests"]`;
there are no markers or custom flags.

## Import order: `tests/conftest.py`

The whole file is one line:

```python
import nnsight  # noqa: F401  nnsight before any transformers submodule, in every test process
```

On this stack, importing a `transformers.models.*.modeling_*` module before
nnsight segfaults at import ([transformers-compat.md](transformers-compat.md)).
Test files that import a modeling class (`tests/test_base.py:6-7`,
`tests/test_registry.py:70`) still write `from nnsight import
TransformersModel` first for the reader; the conftest is what makes the
order hold in every process pytest starts.

## `FamilySuite` (`tests/families/suite.py`)

A family's test file subclasses `FamilySuite`, sets the class attributes,
and inherits every method. Two class-scoped fixtures load the checkpoint
once per class: `model` (`suite.py:88-92`), a `StandardizedTransformer` with
`dispatch=True, attn_implementation="eager"` plus `LOAD_KWARGS`, and
`raw_model` (`:93-97`), the same checkpoint as a plain `TransformersModel`.

### Class attributes (`suite.py:137-170`)

| attribute | meaning |
|---|---|
| `REPO` | the pinned tiny checkpoint |
| `FAMILY` | the family module the checkpoint must resolve to |
| `NATIVE` | standard path → native path, usually built by `rows(container, layers, embed, norm, attn=, mlp=, ln1=, ln2=)` (`:58-73`); `LLAMA_ROWS` (`:76`) is the Llama layout |
| `EXPECTED_UNAVAILABLE` | `support()` key → a substring of the reason, for values this checkpoint lacks; a module no block has (OPT's `mlp`) is absent from `support()` rather than unavailable, so it is not listed here |
| `REFUSES_IN_PLACE_QKV` | q/k/v come out of a multi-view op (`split`, `chunk`), so torch refuses an in-place edit (GPT-2, MPT) |
| `ATTENTION_SINK` | the pattern's rows sum to less than one (GPT-OSS) |
| `KV_HEADS_EXPANDED` | keys and values are read already expanded to `num_heads` (DeepSeek, Falcon 40B) |
| `MLP_WIDTH_KEY` | a config key naming the MLP width when it is not `intermediate_size` |
| `LOAD_KWARGS` | extra load arguments (DBRX needs `dtype=torch.float32`) |
| `QUERY_GATED` | `q_proj` yields the query and a gate side by side (Qwen3.5) |
| `ATTENTION_NORM` | the block module whose output enters the attention; `None` when the block input does (OLMo-2/3) |
| `MLP_NORM` | the block module whose output enters the MLP, whatever the family calls it |
| `MLP_NORM_BEFORE_ATTENTION` | the MLP's norm fires before the attention (Falcon 40B norms both inputs up front) |
| `MOE_UNAVAILABLE` | mixture value → a substring of the reason, for the mixture values this checkpoint lacks on its mixture (a mixture without a shared expert needs no entry for `shared_expert_output`) |
| `ROUTER_EXTRA_CLASSES` | router classes beyond the experts: ZAYA's skip class is one more column of `router_logits` |

Three helpers: `mixer(layer)` (`:40-43`) is `self_attn` or, on a linear
block, `linear_attn`; `has_mlp(model)` (`:184-186`) reads `support()`;
`attn_blocks(model)` is the blocks with softmax attention and `attn_block(model)` the first of them, skipping the test when there is none;
`recurrent_mixers(model)` is the family's `RecurrentMixer` classes from its `ENVOYS`;
`expected_values(model)` (`:106-112`) adds the `linear_attn.*` names on a
hybrid and drops `mlp.mlp_output` when no block has an MLP module (OPT). `VALUES`, `INTERIOR` and `LINEAR` (`:20-27`) are the value name sets.

### The test methods, by what they assert

**Names** (`suite.py:115-155`)

| method | asserts |
|---|---|
| `test_family_resolved` | `model.family is FAMILY` |
| `test_standard_names_alias_native_envoys` | every `NATIVE` row: the standard path resolves to an `Envoy`, and it *is* the native envoy |
| `test_no_inner_model_alias` | nothing is bound at `model.model`: the containers are lifted to the root |
| `test_every_layer_is_renamed` | every `layers.0.*` row of `NATIVE` resolves on every block, and there is more than one block |
| `test_envoy_classes` | every block is exactly `FAMILY.Layer`, every softmax attention `FAMILY.Attention`, every MLP `FAMILY.Mlp`, every mixer `FAMILY.LinearAttention`, and each subclasses the component base |
| `test_trace_through_standard_names` | a trace through `mixer(...).output`, `layers[-1].output`, `norm.output`, `lm_head.output` gives `hidden_size` / `vocab_size` widths |

**Availability** (`:158-184`)

| method | asserts |
|---|---|
| `test_support_lists_every_standard_value` | `support()` keys equal `expected_values`; `support(layer=0)` drops the six root keys |
| `test_support_is_what_this_family_expects` | every key in `EXPECTED_UNAVAILABLE` is a non-empty per-block dict whose reasons contain the substring; every other key is `None` |
| `test_support_matches_what_reads` | on block 0, an available name is an `EProperty` on the host's class; a missing module says `no ... module`; an unavailable value raises `Unavailable` matching the reason |

**Boundary values** (`:188-269`)

| method | asserts |
|---|---|
| `test_layer_output_is_the_tensor` | `layer_output` equals `.output` (or its first element), width `hidden_size` |
| `test_every_value_reads_a_tensor_on_every_layer` | `attention_output`, `mlp_output`, `layer_output` read a `hidden_size`-wide tensor on every block, one trace |
| `test_contribution_identity` | `input + attention_output + mlp_output == layer_output` on every block, at 8 ulp in float32 (skipped without an MLP) |
| `test_sublayer_inputs_are_the_normed_stream` | `self_attn.input` equals `ATTENTION_NORM`'s output (or the block input); `mlp.input` equals `MLP_NORM`'s output; reads ordered by the forward, one norm shared on a parallel block |
| `test_renamed_model_equals_raw_model` | block 0's output and `lm_head.output` equal the raw `TransformersModel`'s |
| `test_boundary_writes_land` | assigning `attention_output * 0`, `mlp_output * 0`, `layer_output * 0`, and zeroing `layer_output[:]` in place, each moves the logits |

**The attention pattern** (`:273-332`)

| method | asserts |
|---|---|
| `test_probabilities_are_a_pattern` | shape `[batch, num_heads, seq, seq]`, dtype of `lm_head.weight`, rows sum to one at 8 ulp (or lie in (0, 1) with `ATTENTION_SINK`), lower-triangular |
| `test_pattern_across_layers_and_traces` | first and last attention blocks have the same shape and differ; two traces of the first agree exactly |
| `test_written_pattern_moves_the_logits` | assigning a random pattern and zeroing head 0 in place both move the logits (a read can be causally inert) |
| `test_every_source_value_resolves_on_every_layer` | every available value on `FAMILY.Attention` whose path is inside a forward (`inside_forward()`) reads a tensor on every attention block, one trace per value |

**The attention interior** (`:336-396`)

| method | asserts |
|---|---|
| `test_interior_shapes` | q `[b, heads, s, qk_head_dim]`, k/v `[b, kv_heads, s, ·]`, scores and pattern `[b, heads, s, s]`, head outputs `[b, s, heads, ·]`; `pattern_from_scores(scores)` equals the pattern (overridable: GPT-OSS adds the sink column) |
| `test_interior_writes_are_causal` | zeroing each of the five interior values moves the logits; with head outputs zeroed the contribution is the same at every position |
| `test_interior_in_place_edits` | zeroing scores and head outputs in place moves the logits; queries too, or raise `RuntimeError` matching `view` with `REFUSES_IN_PLACE_QKV` |

**Methods over the values** (`:400-441`)

| method | asserts |
|---|---|
| `test_skip_layers_hands_the_stream_straight_through` | skipping blocks 1..last leaves the last `layer_output` equal to block 0's, and the logits equal `project_on_vocab` of it |
| `test_skip_layers_with_a_given_stream` | `skip_with=zeros` makes block 0's output and block 1's input zero |
| `test_steer` | the last position moves by `2 * vector`, the others are untouched, the logits move |
| `test_project_on_vocab_is_the_logit_lens` | on the last block's output it equals `logits`; `get_topk_closest_tokens(k=3)` returns one dict of three token → probability entries summing to at most one |

**Layouts** (`:445-485`)

| method | asserts |
|---|---|
| `test_values_match_their_annotations` | every value with a `layout`, on the root, a block, its attention, its MLP and one mixer, is an instance of its layout alias (`Residual`, `Pattern`, ... from `nnterp.components`) and each named axis matches `axis_sizes` (`hidden_size`, `num_heads`, `head_k_dim`, `streams` = the text config's `hc_mult` where it has one, ...); one value per trace; at least 11 checked |

**The input** (`:489-514`)

| method | asserts |
|---|---|
| `test_input_accessors` | `input_ids`, `input_size`, `attention_mask` are `[1, n]` for the prompt's `n` tokens, the mask all ones, `token_embeddings` `n` long, and the three names print in the repr |
| `test_assigning_input_ids_runs_other_ids` | assigning another prompt's ids and mask reproduces that prompt's logits; assigning `input_size` raises `AttributeError` |

**The root** (`:518-580`)

| method | asserts |
|---|---|
| `test_logits_are_the_models_output` | `logits` equals `model.output.logits`, and `model.project_on_vocab(layers[-1].layer_output)` (the softcap, or the family's own step) |
| `test_assigning_logits_replaces_the_result` | `logits = logits * 0` zeros `tracer.result.logits` |
| `test_token_embeddings_are_the_embedding_output` | equals `embed_tokens.output`; assigning it moves the logits |
| `test_next_token_probs` | equals `logits[:, -1].softmax(-1)`, prints in the repr, assignment raises `AttributeError` |
| `test_sizes_match_the_model` | `num_layers`, `num_heads` against the pattern, `hidden_size` against the stream, `1 <= num_kv_heads <= num_heads`, `q_proj`/`o_proj` widths, and the MLP width appears in some block parameter's shape |
| `test_per_module_sizes_match_each_block` | each attention block's `self_attn.num_heads`, `num_kv_heads`, `head_dim`, `qk_head_dim` against its `q_proj`/`k_proj`/`v_proj`/output projection widths, and against the queries, keys, values, pattern and head outputs of one block of each kind (module type and parameter shapes); each `mlp.intermediate_size` is an axis of its own (routed experts') weights, the root's on a dense block, `MLP_WIDTH_KEY` on the first MLP block where set |
| `test_repr_lists_the_values` | the block repr lists `layer_output`, `attention_output`, `attention_probabilities`, `attention_queries`, `attention_head_outputs`, and `mlp_output` when there is an MLP |

## How a family file adds specifics

Most files are the class attributes alone (`tests/families/test_gpt2.py:8-12`
adds `REFUSES_IN_PLACE_QKV = True` and one test that flipping
`reorder_and_upcast_attn` on the config makes the interface unavailable,
`:14-22`). Two files show the two other things a family file does.

`tests/families/test_falcon.py` has **three classes for one family**: the 7B
layout, the 7B layout with alibi, and the 40B layout:

```python
class TestFalcon(FamilySuite):                       # 7B layout
    REPO = "Rocketknight1/tiny-random-falcon-7b"
    FAMILY = falcon
    NATIVE = rows("transformer", "h", "word_embeddings", "ln_f", attn="self_attention", ln2=None)
    MLP_NORM = "input_layernorm"                     # parallel: one norm feeds both sublayers
    # + test_mlp_output_is_a_copy_the_block_does_not_touch, test_in_place_mlp_edit_reaches_the_model_through_the_transform,
    #   test_values_bind_before_the_rotary

class TestFalconAlibi(FamilySuite):                  # 7B layout, config.alibi = true
    REPO = _alibi_checkpoint()                       # the 7B snapshot with alibi written into config.json
    FAMILY = falcon
    NATIVE = rows("transformer", "h", "word_embeddings", "ln_f", attn="self_attention", ln2=None)
    MLP_NORM = "input_layernorm"
    # + test_alibi_branch

class TestFalcon40B(FamilySuite):                    # new_decoder_architecture
    REPO = "Rocketknight1/tiny-random-falcon-40b"
    FAMILY = falcon
    NATIVE = rows("transformer", "h", "word_embeddings", "ln_f", attn="self_attention", ln1=None, ln2=None)
    KV_HEADS_EXPANDED = True
    ATTENTION_NORM = "ln_attn"
    MLP_NORM = "ln_mlp"
    MLP_NORM_BEFORE_ATTENTION = True
```

(`test_falcon.py:14-89`). The three extra 7B tests pin what the family module
says: `mlp_output` is a copy equal to `mlp.output - attention_output`
(`:20-26`), an in-place edit on the copy still moves the logits and the
user's tensor stays what they made it (`:28-36`), and `attention_values` must
be read before `attention_queries` in one trace (`:38-44`). The alibi class
runs the whole suite on the other attention branch: `_alibi_checkpoint()`
(`:47-57`) symlinks the 7B snapshot's files into a temp dir and writes
`alibi: true` into its `config.json`, the way OLMo-3's test patches its
checkpoint, and `test_alibi_branch` (`:68-73`) checks the pattern equals the
`self_attention_dropout_0` output.

`tests/families/test_olmo3.py` **patches a checkpoint**: the tiny OLMo-3
checkpoint's flat `rope_parameters` does not parse for a model with
`layer_types`, so `_patched_checkpoint()` (`test_olmo3.py:14-32`) symlinks
the snapshot's files into a temp dir, rewrites only `config.json` into the
per-layer-type form, and `REPO` is that directory (`:36`). The family itself needs nothing. Its own
test checks the contribution is the post-attention norm's output (`:42-46`).
`tests/families/test_gemma4_text.py` patches weights, not only the config: the
tiny text checkpoint's per-layer embedding table has 99 rows, so `_ple_checkpoint()`
tiles it to the full vocabulary into a copy under the temp directory, written once
per snapshot. Its `Gemma4Suite` overrides `test_contribution_identity` for the
block's `* layer_scalar` and third term, adds `per_layer_output` to
`expected_values`, and overrides `test_skip_layers_with_a_given_stream` on a
KV-sharing checkpoint; `test_gemma4_unified_text.py` imports that suite.
`tests/families/test_deepseek_v4.py` overrides `test_contribution_identity` with the
hyper-connection block's stream formula (checked on every block), adds the four stream
weights to `expected_values`, and gives `pattern_from_scores` GPT-OSS's sink column; the
suite's key-length checks run on block 0 (a sliding block) with a prompt shorter than any
compression rate, so the family's own test covers the longer keys of the compressed blocks.
The hybrid files (`test_qwen3_next.py`, `test_qwen3_5_text.py`,
`test_qwen3_5_moe_text.py`) are one file with three headers: they override
`test_every_layer_is_renamed` for the per-block `self_attn`/`linear_attn`
split and add the DeltaNet tests
([recurrent-mixer-internals.md](recurrent-mixer-internals.md)).
The Mamba-1 files (`test_mamba.py`, `test_falcon_mamba.py`, `test_jamba.py`)
mix `SelectiveScanSuite` (`tests/families/scan_suite.py`) in before
`FamilySuite`: a class-scoped autouse fixture routes the family to the
pure-torch kernels for the whole class, and its tests check the values
against the recurrence recomputed from them, the writes, the decode step and
the per-token state on `SCAN_BLOCK`, and meta-build a real checkpoint's config
(`REAL`). On a model with no softmax attention (Mamba), `attn_block(model)`
skips the pattern and interior tests and `expected_values` drops the
`self_attn.*` names.
A multimodal wrapper runs `FamilySuite` loaded with `LOAD_KWARGS = {"task":
"image-text-to-text"}` (a subclass of the family's text class with the wrapper's
`NATIVE` rows; `expected_values` adds `image_token_mask` and `image_features`
where the load has a processor), and its vision side runs `VisionSuite`
(`tests/families/vision_suite.py`: the tower names and identity, the scatter
`layers[0].input[image_token_mask] == image_features`, causal edits, no image values
on a text-only checkpoint). `WrapperSuite`, in the same file, checks the text names
on a wrapper built from a family's tiny text config where no tiny wrapper exists
([vision-design.md](vision-design.md)).

## The root tests

`tests/test_registry.py` (13 tests): every module under `nnterp/families/` is
named after its single `MODEL_TYPES` entry and there are at least 31
(`:15-20`); `import nnterp` pulls in no `transformers.models.*.modeling_*`
module and `lookup("gpt2")` imports only that family, checked in a
subprocess (`:23-36`); an unknown `model_type` raises `UnsupportedFamily`
(`:39-41`); `register` adds and overrides, and `lookup` returns it (`:44-51`);
a preloaded `nn.Module` uses its own config (`:54-58`); a user `rename`
merges over the family's (`:61-64`); a user `envoys` type key replaces the
family's (`:67-77`); the remote key is `TransformersModel` (`:80-85`); a
default load keeps the checkpoint's attention and `support()` names `eager`
(`:88-92`); a family registered with only `MODEL_TYPES`, `RENAME` and `ENVOYS`
answers `support()` with every block value (`:95-103`); a value on an
`Attention` subclass passed through `envoys=` is listed by `support()` and
`support(layer=0)` as `self_attn.<name>`, and a load without it does not list
it (`:106-118`); a family registered with `hidden_size=lambda model: 999`
answers `model.hidden_size` with 999 while `num_heads` keeps the root's rule,
and a plain load reads `config.hidden_size` (`:121-131`); assigning a size
(`model.hidden_size = 5`) raises `AttributeError` naming `def hidden_size`
(`:134-137`).

`tests/test_base.py` (2 tests): the base `Layer` on a plain
`TransformersModel` over GPT-J unwraps and rewraps a tuple block output
(`:13-36`); an `unavailable(...)` marker is listed in the repr with its
reason, reported by `support()` at both levels, raises `Unavailable` on read,
and `hasattr` raises too (`:39-52`).

`tests/test_prompt_utils.py` (7 tests, GPT-2 with `tokenizer_kwargs`
setting left padding and a pad token): `tokenizer_kwargs` are applied;
`get_first_tokens` returns one or two deduplicated ids per word and uses the
`add_prefix_false_tokenizer`; the hacky (pear) implementation agrees;
`Prompt.from_strings` names targets and detects collisions;
`get_target_probs` reduces per target and per layer; `Prompt.run` with
`next_token_probs_unsqueeze` gives `[1, 1, vocab]` distributions;
`run_prompts` batched equals one by one and refuses mismatched targets.

`tests/test_nnsight_utils.py` (6 tests, same fixture): `get_token_activations`
is `[num_layers, num_prompts, hidden]` on the CPU and equals
`layer_output[:, -1]`; it reads inside a caller's tracer; a positive index
needs right padding; batched and one-at-a-time collection agree; the
session collector agrees; `compute_next_token_probs` rows sum to one.

## Gotchas

- Names bound inside a `with model.trace(...)` block do not survive it;
  the suite pre-binds containers (`read = {}`, `saved = None`) outside the
  block (`suite.py:198`, `:472`). A plain list built inside the block needs
  `.save()` too.
- Reads in one trace follow the forward: the suite reads one interior value
  per trace (`:327`, `:340`) because families bind them at different points.
- The class-scoped `model` fixture is shared by every test in the class;
  a test that needs another config loads a patched copy of the checkpoint as
  its own class (`_alibi_checkpoint()`, `test_falcon.py:47-57`;
  `_patched_checkpoint()`, `test_olmo3.py:14-32`) rather than mutating the
  fixture's.
- `route_delta_rule(..., "recurrent")` (`route_kernels(..., "torch")`) is
  process-wide; the hybrid tests load a second model after routing and restore
  `"chunked"` in a `finally`.
- pytest imports `tests/families/suite.py` as the bare module `suite`
  (`from suite import FamilySuite`); do not name another test helper that.
- Run from the repository root: `git rev-parse --show-toplevel` is
  the nnterp checkout, and pytest's `testpaths` is relative to it.

## Related

- [contributing.md](contributing.md) — the workflow around the suite
- [transformers-compat.md](transformers-compat.md) — the tests that guard op names
- [recurrent-mixer-internals.md](recurrent-mixer-internals.md) — what the hybrid tests pin
- nnsight `docs/developing/testing.md` — the suite of the library underneath
