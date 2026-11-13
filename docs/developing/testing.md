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
pip install -e ".[test,models]"          # nnterp editable, with the test and models extras
export HF_HUB_OFFLINE=1
pytest                                   # the whole suite
pytest -n 8                              # the same on eight pytest-xdist workers
pytest tests/families/test_gpt2.py -q    # one family
pytest tests/test_registry.py -q         # one test spawns a subprocess
pytest "tests/families/test_gpt2.py::TestGPT2::test_contribution_identity" -q -x   # one test, stop on first failure
```

`HF_HUB_OFFLINE=1` keeps a run from touching the Hub once the pinned checkpoints are
cached; a first run without it downloads them. `pyproject.toml` sets `testpaths =
["tests"]` and `addopts = "--dist loadgroup"`, so under `-n` a class marked
`xdist_group` stays on one worker. The classes that load a wrapper whose processor imports
`torchvision` (the Qwen ViT's, Gemma 4's, NemotronH Omni's) error without it, and
SmolVLM's processor needs `num2words`; neither is in an extra.

## Import order: `tests/conftest.py`

It imports nnsight first and gives each xdist worker one torch thread.
A test file that imports a modeling class at its top (`tests/test_base.py`) still
writes `from nnsight import TransformersModel` first for the reader; the
conftest is what makes the order hold in every process pytest starts.

## `FamilySuite` (`tests/families/suite.py`)

A family's test file subclasses `FamilySuite`, sets the class attributes,
and inherits every method. Two class-scoped fixtures load the checkpoint
once per class: `model` (`suite.py`), a `StandardizedTransformer` with
`dispatch=True, attn_implementation="eager"` plus `LOAD_KWARGS`, and
`raw_model`, the same checkpoint as a plain `TransformersModel`.

### Class attributes (`FamilySuite`, `suite.py`)

| attribute | meaning |
|---|---|
| `REPO` | the pinned tiny checkpoint, or `built(patched_checkpoint, repo, ...)`: a copy of it in a config branch no pinned tiny takes (below) |
| `FAMILY` | the family module the checkpoint must resolve to |
| `NATIVE` | standard path → native path, usually built by `rows(container, layers, embed, norm, attn=, mlp=, ln1=, ln2=)`; `LLAMA_ROWS` is the Llama layout |
| `EXPECTED_UNAVAILABLE` | `support()` key → a substring of the reason, for values this checkpoint lacks; a module no block has (OPT's `mlp`) is absent from `support()` rather than unavailable, so it is not listed here |
| `REFUSES_IN_PLACE_QKV` | the queries come out of a multi-view op (`split`, `chunk`), so torch refuses an in-place edit of them (GPT-2, GPT-BigCode, MPT) |
| `ATTENTION_SINK` | the pattern's rows sum to less than one (GPT-OSS) |
| `KV_HEADS_EXPANDED` | keys and values are read already expanded to `num_heads` (DeepSeek, Falcon 40B) |
| `MLP_WIDTH_KEY` | a config key naming a mixture's MLP width when it is not `intermediate_size` (the experts'), read where the first block with an MLP is a mixture (`mlp_width_key(model)`), so a checkpoint whose first MLP is dense (ERNIE-4.5-21B, Nemotron-3-Nano-4B) takes the plain rule |
| `LOAD_KWARGS` | extra load arguments (DBRX needs `dtype=torch.float32`) |
| `QUERY_GATED` | `q_proj` yields the query and a gate side by side (Qwen3.5) |
| `ATTENTION_NORM` | the block module whose output enters the attention; `None` when the block input does (OLMo-2/3) |
| `MLP_NORM` | the block module whose output enters the MLP, whatever the family calls it |
| `MLP_NORM_BEFORE_ATTENTION` | the MLP's norm fires before the attention (Falcon 40B norms both inputs up front) |
| `MOE_UNAVAILABLE` | mixture value → a substring of the reason, for the mixture values this checkpoint lacks on its mixture (a mixture without a shared expert needs no entry for `shared_expert_output`) |
| `ROUTER_EXTRA_CLASSES` | router classes beyond the experts: ZAYA's skip class is one more column of `router_logits` |

Methods a family file overrides the way it sets an attribute: `pattern_from_scores`
(what the softmax makes of the scores; GPT-OSS adds its sink column);
`rotated_as_served(attn, name, rotated, positions)` (what the attention receives given `rotate` of a
value before the rotary at those positions; the identity, HunYuan's and Llama 4's post-rotary norms, JetMoE's
tiling of the keys, Ministral 3's query scale by position); `native(model)` (`NATIVE` for the checkpoint
loaded: a hybrid names its first block of each kind from the config's `layer_types`, so the class runs on a
trained checkpoint whose blocks sit elsewhere); `mixture_output(model, host)` (what `routed_output +
shared_expert_output` equals: JetMoE subtracts its mixture's bias, the Granite families divide by the
multiplier); and `routed_scale(host)` / `routed_from_experts(host, summed)` (what a sum of `expert_outputs`
adds to `routed_output`: Laguna's `routed_scaling_factor`, Nemotron-H's `fc2_latent_proj` under
`moe_latent_size`). `model_size(host, axis, name)` takes the value's name, so JetMoE's file
says its keys before the rotary are `num_kv_heads` wide where `KV_HEADS_EXPANDED` holds for the rest.
A test that compares with a fact of the checkpoint (a block index, a multiplier, a width) reads it from the
config or the module tree rather than writing the tiny checkpoint's value, so the class also runs on a
trained checkpoint of the family; `blocks(model, kind)` lists the blocks the text config's `layer_types`
names `kind`.

Helpers: `mixer(layer)` is `self_attn` or, on a linear
block, `linear_attn`; `has_mlp(model)` reads `support()`;
`attn_blocks(model)` is the blocks with softmax attention and `attn_block(model)` the first of them, skipping the test when there is none;
`recurrent_mixer(family)` is the family's `RecurrentMixer` class from its `ENVOYS`;
`expected_values(model)` adds the `linear_attn.*` names on a
hybrid and drops `mlp.mlp_output` when no block has an MLP module (OPT). `VALUES`, `INTERIOR` and `LINEAR` are the value name sets.

### The test methods, by what they assert

**Names** (`suite.py`)

| method | asserts |
|---|---|
| `test_family_resolved` | `model.family is FAMILY` |
| `test_standard_names_alias_native_envoys` | every `NATIVE` row: the standard path resolves to an `Envoy`, and it *is* the native envoy |
| `test_no_inner_model_alias` | nothing is bound at `model.model`: the containers are lifted to the root |
| `test_every_layer_is_renamed` | every `layers.0.*` row of `native(model)` resolves on every block |
| `test_envoy_classes` | every block is exactly `FAMILY.Layer`, every softmax attention `FAMILY.Attention`, every MLP `FAMILY.Mlp` (or `FAMILY.Moe`), every mixer the `RecurrentMixer` subclass the family's `ENVOYS` keys, and each subclasses the component base |
| `test_trace_through_standard_names` | a trace through `mixer(...).output`, `layers[-1].output`, `norm.output`, `lm_head.output` gives `hidden_size` / `vocab_size` widths |

**Availability**

| method | asserts |
|---|---|
| `test_support_lists_every_standard_value` | `support()` keys equal `expected_values`; `support(layer=0)` drops the eight root keys |
| `test_support_is_what_this_family_expects` | every key in `EXPECTED_UNAVAILABLE` is a non-empty per-block dict whose reasons contain the substring; `mlp_activation` and `mlp_neurons`, where not listed, are unavailable exactly on the blocks whose MLP is a routed mixture ("routed experts") or that have none ("no mlp module"); every other key is `None` |
| `test_support_matches_what_reads` | on block 0, an available name is an `EProperty` on the host's class; a missing module says `no ... module`; an unavailable value raises `Unavailable` matching the reason |

**Boundary values**

| method | asserts |
|---|---|
| `test_layer_output_is_the_tensor` | `layer_output` equals `.output` (or its first element), width `hidden_size` |
| `test_every_value_reads_a_tensor_on_every_layer` | `attention_output`, `mlp_output`, `layer_output` read a `hidden_size`-wide tensor on every block, one trace |
| `test_contribution_identity` | `input + attention_output + mlp_output == layer_output` on every block, at 8 ulp in float32 (skipped without an MLP) |
| `test_sublayer_inputs_are_the_normed_stream` | `self_attn.input` equals `ATTENTION_NORM`'s output (or the block input); `mlp.input` equals `MLP_NORM`'s output; reads ordered by the forward, one norm shared on a parallel block |
| `test_renamed_model_equals_raw_model` | block 0's output and `lm_head.output` equal the raw `TransformersModel`'s |
| `test_boundary_writes_land` | assigning `attention_output * 0`, `mlp_output * 0`, `layer_output * 0`, and zeroing `layer_output[:]` in place, each moves the logits |
| `test_assigning_layer_mid_replaces_the_stream` | on the first block serving `layer_mid`, an assignment gives the in-place write's `layer_output` bit for bit, under one invoke and under two that both write; `layer_output == new + mlp_output` where the clean block sums plainly; the gradients of a learned scale and shift of a cloned read reach the assigned value as through the in-place write |

**The attention pattern**

| method | asserts |
|---|---|
| `test_probabilities_are_a_pattern` | shape `[batch, num_heads, seq, seq]`, dtype of `lm_head.weight`, rows sum to one at 8 ulp (or lie in (0, 1) with `ATTENTION_SINK`), lower-triangular |
| `test_pattern_across_layers_and_traces` | first and last attention blocks have the same shape and differ; two traces of the first agree exactly |
| `test_written_pattern_moves_the_logits` | assigning a random pattern and zeroing head 0 in place both move the logits (a read can be causally inert) |
| `test_every_source_value_resolves_on_every_layer` | every available value on `FAMILY.Attention` whose path is inside a forward (`inside_forward()`) reads a tensor on every attention block, one trace per value |

**The attention interior**

| method | asserts |
|---|---|
| `test_interior_shapes` | q `[b, num_heads, s, qk_head_dim]`, k/v `[b, num_kv_heads, s, ·]`, scores and pattern `[b, num_heads, s, s]`, head outputs `[b, s, num_heads, ·]`; `pattern_from_scores(scores)` equals the pattern (overridable: GPT-OSS adds the sink column) |
| `test_interior_writes_are_causal` | zeroing each of the five interior values moves the logits; with head outputs zeroed the contribution is the same at every position |
| `test_interior_in_place_edits` | zeroing scores and head outputs in place moves the logits; queries too, or raise `RuntimeError` matching `view` with `REFUSES_IN_PLACE_QKV` |
| `test_interior_per_invoke` | on a family that reads the interior off its own arithmetic (no `interface`), under two invokes each interior value is that invoke's row of one invoke of both, and an assignment in the first leaves the second's logits as they were (XGLM's keys and values are not per invoke: `TestXGLM` checks the other four and marks those two `xfail`) |

**The rotary embedding**

| method | asserts |
|---|---|
| `test_rotate_gives_the_rotated_queries_and_keys` | on the first block with a rotary, `self_attn.rotate(prerope, arange(seq))` equals `attention_queries` / `attention_keys` bit for bit, in their dtype, and has their shape where the two layouts are the same name (`rotated_as_served` first: a family's test file says what its block does after the rotary, HunYuan's norms, Llama 4's `qk_norm`, JetMoE's tiling); a rotated value narrower than the served one (latent attention's shared rope key) is its trailing channels on every head |
| `test_a_prerope_write_is_rotated_for_its_position` | assigning position 0's prerope vector at the last position leaves the rotated value at every other position bit-identical and makes the last one `rotate(vector, [last])` |
| `test_rotate_says_why_a_block_has_no_rotary` | on every block whose `no_rotary()` gives a reason, `support()` gives the same one and `rotate` raises `Unavailable` with it |
| `test_order_ranks_the_prerope_values_first` | `model.order(i)` and `model.rank` put both values before the rotated ones |

`NO_ROTARY` (`suite.py`) is the `EXPECTED_UNAVAILABLE` entry of a family without a rotary
(GPT-2, OPT, BLOOM, ...) or with NoPE blocks (Cohere 2, SmolLM3, ...); `TOWER_NO_ROTARY` is
what every vision tower reports for the two values, which the suite expects on a wrapper's
`vision.*` rows without a family saying so.

**Methods over the values**

| method | asserts |
|---|---|
| `test_skip_layers_hands_the_stream_straight_through` | skipping blocks 1..last leaves the last `layer_output` equal to block 0's, and the logits equal `project_on_vocab` of it; skipped on a one-block checkpoint |
| `test_skip_layers_with_a_given_stream` | `skip_with=zeros` makes block 0's output and block 1's input zero; skipped on a one-block checkpoint |
| `test_steer` | the last position moves by `2 * vector`, the others are untouched, the logits move |
| `test_project_on_vocab_is_the_logit_lens` | on the last block's output it equals `logits`; `get_topk_closest_tokens(k=3)` returns one dict of three token → probability entries summing to at most one |

**Calling the attention function again** (`self_attn.interface`; skipped where no block serves it, whose reason `test_support_is_what_this_family_expects` checks through `EXPECTED_UNAVAILABLE`)

| method | asserts |
|---|---|
| `test_interface_reproduces_heads_and_pattern` | on every attention block, in one trace, `interface(attention_queries, attention_keys, attention_values)` equals `attention_head_outputs` and `attention_probabilities` bit for bit |
| `test_interface_on_a_padded_batch` | the same on a padded batch of `PROMPTS` in one invoke (pad rows included, NaN equal to NaN), and to rounding under two invokes; skipped without a pad token |
| `test_interface_patch_reaches_its_group_alone` | read once and called twice, the call on keys and values of key/value head 0 patched moves exactly the query heads `h < num_heads // num_kv_heads`; written into the last row of `attention_head_outputs`, it moves the attention module's output at the last position and nowhere else |
| `test_interface_on_decode_steps` | under `generate` and `tracer.iter[:2]`, the call reproduces both values on the prompt and on a decode step (one query row) |

`VisionSuite.test_tower_interface_reproduces_heads_and_pattern` checks the same on a tower block, and
`tests/test_tensor_parallel.py` reads `interface` and patches through it across two CPU ranks.

**The weights per head** (`value_weight`, `output_weight`, `attention_input_gain`, `attention_output_gain`; the first
two skipped where no attention block serves them, whose reasons are checked against `EXPECTED_UNAVAILABLE`)

| method | asserts |
|---|---|
| `test_value_weight_reproduces_the_values` | on the first block serving them, for every key/value head, `value_weight(kv) @ self_attn.input` plus the head's bias (the values with the attention's input zeroed) is `attention_values[:, kv]`, by `torch.testing.assert_close` at the dtype's default tolerance (a matmul over a slice of rows does not sum in the whole projection's order, so not bit for bit); `value_weight(num_kv_heads)` raises `IndexError` |
| `test_output_weight_reproduces_the_projection` | `sum_j output_weight(j) @ attention_premix[:, :, j]` plus the bias (the output with the heads zeroed) is `o_proj.output`, at the default tolerance on both divided by the output's largest magnitude (an element that cancels to near zero keeps the rounding of the terms it sums: DeepSeek-V3) |
| `test_attention_input_gain` | on block 0, with the norm's weight set to a ramp for the test (`ramped`: a tiny checkpoint's is often all ones), `attention_norm_output` is `gain * n(layer_input)` plus the norm's bias (`normalized`), `n` the RMS (or, on a norm that takes a constant to its bias, centred) normalization with the norm's own eps, and the gain is float32 and detached; ones on a post-norm block; `Unavailable` with `attention_norm_output`'s reason otherwise |
| `test_attention_output_gain` | on the first attention block, where `attention_output` is read at a norm after the attention, that norm's output is `gain * n(o_proj.output)` plus its bias, ramped and normalized as above; ones where it is read at the attention's own output, at an operation, or at an identity (HyperCLOVA X without `use_post_norm`); `Unavailable` with `attention_output`'s reason where that is refused |
| `test_head_weights_say_why_a_block_has_none` | each method is served on every attention block, or raises `Unavailable` containing `EXPECTED_UNAVAILABLE["self_attn.value_weight"]` / `["self_attn.output_weight"]` on every one; `no_head_weights()` contains the output weight's reason where both are refused, and is `None` otherwise |

`"self_attn.value_weight"` and `"self_attn.output_weight"` are keys `support()` does not have (the weights are
methods), so `test_support_is_what_this_family_expects` never meets them; latent attention, Gemma-4 and ZAYA name
the first only, DeepSeek-V4, JetMoE and the default family both. Layouts no tiny checkpoint has run the value
test on random weights (`random_weights(repo, **config)`): GPT-BigCode without multi-query, CodeGen with two
heads per `mp_num` part, and Phi-3 under grouped-query attention; Falcon-RW's runs it in `TestFalconRW`.
`test_gemma2.py` checks that
on a bf16 load both gains are `1 + weight` in float32, bit for bit. `tests/families/test_llama.py` runs both
weight tests on the tiny Llama under a throwaway LoRA on `v_proj` and `o_proj` (`peft=` at load, B not zero)
with the adapter enabled, disabled, merged, and disabled while merged, and checks the merged weight is
`value_weight`'s bit for bit.

**Layouts**

| method | asserts |
|---|---|
| `test_values_match_their_annotations` | every value with a `layout`, on the root, a block, its attention, its MLP and one mixer, is an instance of its layout alias (`Residual`, `Pattern`, ... from `nnterp.components`) and each named axis matches `axis_sizes` (`hidden_size`, `num_heads`, `head_k_dim`, `streams` = the text config's `hc_mult` where it has one, ...); one value per trace; at least 11 checked |

**The input**

| method | asserts |
|---|---|
| `test_input_accessors` | `input_ids`, `input_size`, `attention_mask` are `[1, n]` for the prompt's `n` tokens, the mask all ones, `token_embeddings` `n` long, and the three names print in the repr |
| `test_assigning_input_ids_runs_other_ids` | assigning another prompt's ids and mask reproduces that prompt's logits; assigning `input_size` raises `AttributeError` |

**The root**

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

Most files are the class attributes alone (`TestGPT2`,
`tests/families/test_gpt2.py`, adds `REFUSES_IN_PLACE_QKV = True` and one
test that flipping `reorder_and_upcast_attn` on the config makes the
interface unavailable,
`TestGPT2.test_reorder_and_upcast_makes_the_interface_unavailable`). Two
files show the two other things a family file does.

`tests/families/test_falcon.py` has **five classes for one family**: the 7B
layout, the 7B layout with alibi, the 40B layout, and Falcon-RW's and Falcon-11B's
layouts patched into copies of the 7B and 40B checkpoints. The first three:

```python
class TestFalcon(FamilySuite):                       # 7B layout
    REPO = "Rocketknight1/tiny-random-falcon-7b"
    FAMILY = falcon
    NATIVE = rows("transformer", "h", "word_embeddings", "ln_f", attn="self_attention", ln2=None)
    MLP_NORM = "input_layernorm"                     # parallel: one norm feeds both sublayers
    # + test_mlp_output_is_a_copy_the_block_does_not_touch, test_in_place_mlp_edit_reaches_the_model_through_the_transform,
    #   test_values_bind_before_the_rotary

class TestFalconAlibi(FamilySuite):                  # 7B layout, config.alibi = true
    REPO = built(_alibi_checkpoint)                  # the 7B snapshot with alibi written into config.json
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

(`test_falcon.py`). The extra 7B tests pin what the family module
says: `mlp_output` is a copy equal to `mlp.output - attention_output`
(`TestFalcon.test_mlp_output_is_a_copy_the_block_does_not_touch`), an
in-place edit on the copy still moves the logits and the user's tensor
stays what they made it
(`TestFalcon.test_in_place_mlp_edit_reaches_the_model_through_the_transform`),
and `attention_values` must be read before `attention_queries` in one trace
(`TestFalcon.test_values_bind_before_the_rotary`). The alibi class
runs the whole suite on the other attention branch: `_alibi_checkpoint()`
symlinks the 7B snapshot's files into a temp dir and writes
`alibi: true` into its `config.json`, the way OLMo-3's test patches its
checkpoint, and `test_alibi_branch` checks the pattern equals the
`self_attention_dropout_0` output. `TestFalconRW` runs it on falcon-rw-1b's
layout (sequential, alibi, a key and value per head, biased projections).

**A config branch no tiny checkpoint takes** gets a class of its own on a
patched copy of the tiny one, built from its config at test time, never
downloaded: `REPO = built(patched_checkpoint, repo, **config)`
(`suite.py`). `patched_checkpoint` writes `config` over the cached
snapshot's `config.json` (a dict value updates the dict under that key) and
links the weights, or, with `draw=True` or an `init`, draws weights for the new
config with seed 0 and lets `init(model)` set what an initialization leaves
degenerate (`draw_routers`: a router of zeros routes every token alike;
`draw_biases`: zero biases); the copy is written once per change under the
temp dir and shared by the test processes. `cached_snapshot(repo)` resolves a
snapshot in the configured Hugging Face cache (`HF_HOME`, `HF_HUB_CACHE`).
`built(...)` defers the build to the first read of `REPO` (the `model`
fixture), so a snapshot missing from the cache errors that class's tests,
not the module's collection, and a subclass that sets `REPO` to another
checkpoint never builds it. Each class's docstring names the checkpoints that
take its branch.

`tests/families/test_olmo3.py` **patches a checkpoint**: the tiny OLMo-3
checkpoint's flat `rope_parameters` does not parse for a model with
`layer_types`, so `_patched_checkpoint()` (`test_olmo3.py`) symlinks
the snapshot's files into a temp dir, rewrites only `config.json` into the
per-layer-type form, and `REPO` is that directory. The family itself needs nothing. Its own
test checks the contribution is the post-attention norm's output
(`TestOlmo3.test_contributions_are_the_post_norms`).
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
`NATIVE` rows; `expected_values` adds the `vision.*` rows where the load has a
processor), and its vision side runs `VisionSuite`
(`tests/families/vision_suite.py`: the tower names and identity, the tower's values
and `support` rows, the scatter `layers[0].input[vision.image_token_mask] ==
vision.image_features`, causal edits, no `vision` host on a text-only checkpoint or
load; `fix_processor` where a tiny checkpoint's processor disagrees with its model;
`PixtralSuite` for the packed tower). `WrapperSuite`, in the same file, checks the text names
on a wrapper built from a family's tiny text config where no tiny wrapper exists
([vision-design.md](vision-design.md)), and that the logit lens on the wrapper's last block
is the wrapper's own logits (a wrapper can drop a step its text model's causal-LM class
takes: Aya Vision's never applies the `logit_scale` its text config sets, which
`wrapper_of(..., text_overrides=...)` sets away from 1.0 so the test tells the two apart).

## The default family (`tests/families/test_default.py`)

The default family has no checkpoint of its own; it is tested against the
families that do. `load_default(repo)` registers `default` under the
checkpoint's `model_type` for one load. `forced(name)` builds a `FamilySuite`
subclass from a shipped family's suite settings with the default forced on,
so the suite runs on GPT-2, Llama, GPT-NeoX, Phi and OPT, with the
contributions listed unavailable (and OPT's missing `mlp`); `DefaultSuite`
replaces the tests that read a contribution or the logit lens with ones that
assert the refusal. `test_default_reads_what_the_family_reads` loads every
shipped family's checkpoint both ways (but the Mamba-kernel families, whose
mixers need `route_kernels`; mixtures with `experts_implementation=
"batched_mm"`, so the scan runs) and asserts that every value the default
serves is bit-identical to the dedicated family's, and that the logits are;
`REFUSED` lists the families it refuses at load (DBRX's data-dependent
shapes, DeepSeek-V4's parallel streams) and `NO_OP` the pattern reads that
fail loudly on a family whose eager attention forward has no softmax. The rest
pin NanoChat (a `model_type` with no family: the warning, the stream and
logits against the raw model, the one reason for the contributions and the
lens, `steer` and `skip_layers`), Gemma-2's contributions reported
unavailable rather than wrong, and the refusals (RWKV's names,
GPT-NeoX-Japanese's container, Gemma-3n's parallel streams, Inkling's scan
under `grouped_mm`).

## The vLLM suite (`tests/vllm_families/`)

Collected only where `vllm.model_executor` imports and a GPU is visible
(`conftest.py`); everywhere else the directory is ignored. `VLLMFamilySuite`
(`vllm_suite.py`) uses the transformers engine as the oracle:
`transformers_values` runs the checkpoint through `StandardizedTransformer` in
float32 (without gradients, so nothing saved keeps the weights on the card),
and the class-scoped `model` fixture then builds a `StandardizedVLLM` in
float32. The tests hold the engine's boundary values, the module boundaries
and sublayer interiors between them (one value a trace), the final norm's
output, embeddings and logits against the reference within `TOLERANCE` of
each value's largest magnitude;
check the identity on every block, that a read changes no logit, that an
in-place edit, a statement reading twice and an assignment are the same
write, that a value of other rows is refused and the engine survives; and
compare the logits under `steer` and `skip_layers` with transformers'.
Outside a trace they hold `model.rank` / `model.order` to the reference's ranks on the root and
the compared blocks for every value both engines serve (a value the engine lacks raises
`Unavailable` with its reason), and `model.generation_config`'s EOS to the reference's.
`test_lm_head_output` holds `lm_head_output` to `logits` bit for bit, an in-place edit and an
assignment both reaching the logits, where the logits processor applies nothing past the head;
where it softcaps or scales (the family's `UNAVAILABLE` lists `lm_head_output` then), it checks
that the reason names which.
`test_skip_layers` is last: a skip answers for the whole step.

The rotary tests run on the compared blocks that rotate on vLLM. They hold
`attention_queries_prerope` / `attention_keys_prerope` against the reference
within `KERNEL_TOLERANCE` and check that reading them changes no logit
(`test_prerope_values_match_transformers`); check that
`rotate(prerope, arange(tokens))`, called inside the trace, is the
`attention_queries` / `attention_keys` the attention layer receives bit for
bit, and that `rotate` outside a trace raises
(`test_rotate_gives_the_rotated_queries_and_keys`); that position 0's vector
written at the last position before the rotary leaves every other position's
rotated value bit-identical, makes the last one `rotate(vector, [last])`, and
moves the logits as the same write does on the reference
(`test_a_prerope_write_is_rotated_for_its_position`, on the reference's
`rope` block); that on decode step `k` the keys before the rotary, rotated at
`tokens + k - 1`, are that step's keys (`test_rotate_on_decode_steps`); and
that `support()` and `rotate` give the same reason on a block without a
rotary (`test_rotate_says_why_a_block_has_no_rotary`). `test_head_weights_are_refused` checks that
`no_head_weights()` gives the engine's reason on every block, both methods raise it, and
both gains raise too. A family whose block
does more between its rotary and its attention layer overrides
`rotated_as_served(attn, name, rotated)`, called inside the trace, where the
engine's modules run (HunYuan's `query_layernorm` / `key_layernorm`, Llama 4's
`qk_norm` in vLLM's channel order); a family with a block that applies no
rotary, or that does not serve the values, adds `NO_ROTARY` to its
`UNAVAILABLE`.

```bash
PYTHON=/path/to/python tests/vllm_families/run.sh          # every family, one process each
PYTHON=/path/to/python tests/vllm_families/run.sh llama    # one
```

One process per test class (`run.sh` does it) because each class starts an
engine, and engines sharing a card overrun their memory fractions. The
directory is not named `vllm`: with `tests/` on the path that would be
importable as `vllm`.

A family's file names a checkpoint both engines load (`MEMORY` is its
engine's share of the card): a small real one where one is published, else a
tiny one, or a copy of one resized and written once to the temp directory by
the test (Laguna, MiMo-V2-Flash, Llama 4, Ministral 3, HyperCLOVA X, ...),
because vLLM's attention kernels need a head width of at least 16 (32 in
float32; narrower can hang the engine) and most of its MLPs take SiLU only.
The reference reads the attention's interior one value a trace, since each
transformers family reads its own in its own order. Class attributes say
what a family differs in: `SERVED` (the interior values vLLM serves; `()` on
the latent-attention families), `UNAVAILABLE`, `TOLERANCE` and
`KERNEL_TOLERANCE` (set, with the measurement beside them, where the two
engines' attention kernels differ more than the default allows), `DTYPE`,
`ENGINE` (engine arguments: `language_model_only=True` on the
vision-language wrappers, `hf_overrides` where a checkpoint has to be told
what vLLM misreads) and `REFERENCE` (arguments of the transformers
reference's load: a wrapper's `task`, or a config value set to what vLLM
builds). The latent-attention families' files have two classes, one per
attention backend. Two classes skip on a stock install, with the condition
in their file: Apertus (nnsight's meta build must answer `.cpu().item()`)
and DBRX (vLLM 0.27.1's loader does not find the experts). The checkpoints are
about 250 GB together.

## The root tests

`tests/test_tensor_parallel.py` launches `tests/tp_worker.py` in one
process and under `torch.distributed.run` on two CPU ranks over gloo (no GPU), on
`trl-internal-testing/tiny-Qwen3ForCausalLM` (grouped-query, so the keys' head axis
differs from the queries'): every value that tensor parallelism splits by head or
neuron (the queries and keys before the rotary among them, `rotate` of the queries, and
the head outputs and pattern of `interface`), the boundaries nnsight reassembles, and the
logits after seven edits aimed at the last rank's heads, at a query moved before the
rotary, at a head patched through `interface` and at the neurons around the
boundary between the ranks (one with gradients on) must
match the one-process run, and every rank must agree bit for bit; `attention_input_gain` too, while
`value_weight` and `output_weight` are refused on every rank with `SHARDED_WEIGHTS` and served in one
process (`test_head_weights_are_refused_on_a_split_model`). It skips below transformers 5.19.

`tests/test_registry.py`: every module under `nnterp/families/` is
named after its type, and there are at least 31
(`test_every_family_module_is_named_after_its_model_type`); `import nnterp`
pulls in no `transformers.models.*.modeling_*` module and `lookup("gpt2")`
imports only that family, checked in a subprocess (`test_import_is_lazy`);
an unknown `model_type` falls back to the default family with a warning
(`test_unknown_family_falls_back_to_the_default`); `register` adds and
overrides, and `lookup` returns it
(`test_register_adds_a_family_and_can_override`); a preloaded `nn.Module`
uses its own config (`test_preloaded_module_uses_its_own_config`); a user
`rename` merges over the family's (`test_user_rename_merges_over_family`); a
user `envoys` type key replaces the family's
(`test_user_envoys_merge_over_defaults`); the remote key is
`TransformersModel` (`test_remote_key_names_the_plain_transformers_model`); a
default load keeps the checkpoint's attention and `support()` names `eager`
(`test_default_load_keeps_the_checkpoints_attention`); a family registered
with only `RENAME` and `ENVOYS` answers `support()` with every block value
(`test_register_needs_only_names_and_envoys_for_support`); a value on an
`Attention` subclass passed through `envoys=` is listed by `support()` and
`support(layer=0)` as `self_attn.<name>`, and a load without it does not list
it (`test_custom_value_through_envoys_is_in_support`); a family registered
with `hidden_size=lambda model: 999` answers `model.hidden_size` with 999
while `num_heads` keeps the root's rule, and a plain load reads
`config.hidden_size` (`test_family_defines_a_size_instead_of_the_root`);
assigning a size (`model.hidden_size = 5`) raises `AttributeError` naming
`def hidden_size` (`test_sizes_are_read_only`); `register(module)` with no
types covers the type the module's `__name__` ends in
(`test_register_takes_the_type_from_a_modules_name`);
`register(ns, "my_type", "my_other_type")` covers both
(`test_register_takes_several_types`); `register(ns)` on a `SimpleNamespace`
with no types raises `TypeError` and registers nothing
(`test_register_without_a_name_or_types_raises`). `family=` takes a module
imported from the user's path, aliases and size functions included
(`test_family_passed_as_a_module`), and a `SimpleNamespace` with only
`RENAME` and `ENVOYS` that loads and traces
(`test_family_passed_as_a_namespace`); a namespace spread from
`vars(llama)` with an extra alias and envoy class composes with `rename=`
(`test_family_extends_a_shipped_one`); size and `project_on_vocab` functions
on a passed family win
(`test_family_passed_overrides_a_size_and_project_on_vocab`); none of them
touches `REGISTRY`.

`tests/test_base.py`: the base `Layer` on a plain
`TransformersModel` over GPT-J unwraps and rewraps a tuple block output
(`test_tuple_block_unwrapped_and_rewrapped`); an `unavailable(...)` marker is
listed in the repr with its reason, reported by `support()` at both levels,
raises `Unavailable` on read, and `hasattr` raises too
(`test_unavailable_marker_is_listed_and_raises`).

`tests/test_prompt_utils.py` (GPT-2 with `tokenizer_kwargs`
setting left padding and a pad token): `tokenizer_kwargs` are applied;
`get_first_tokens` returns one or two deduplicated ids per word and uses the
`add_prefix_false_tokenizer`; the hacky (pear) implementation agrees;
`Prompt.from_strings` names targets and detects collisions;
`get_target_probs` reduces per target and per layer; `Prompt.run` with
`next_token_probs_unsqueeze` gives `[1, 1, vocab_size]` distributions;
`run_prompts` batched equals one by one and refuses mismatched targets.

`tests/test_nnsight_utils.py` (same fixture): `get_token_activations`
is `[num_layers, num_prompts, hidden_size]` on the CPU and equals
`layer_output[:, -1]`; it reads inside a caller's tracer; a positive index
needs right padding; batched and one-at-a-time collection agree; the
session collector agrees; `compute_next_token_probs` rows sum to one.

## Gotchas

- Names bound inside a `with model.trace(...)` block do not survive it;
  the suite pre-binds containers (`read = {}`, `saved = None`) outside the
  block (`FamilySuite.test_every_value_reads_a_tensor_on_every_layer`,
  `FamilySuite.test_values_match_their_annotations`, `suite.py`). A list first
  bound inside the block is gone after it even when its elements are saved
  (`xs = [v.save() for ...]` and `ys = []; ys.append(v.save())` both raise
  `NameError` after the block); bind it before the block, or save the list
  itself (`nnsight.save([...])`).
- Reads in one trace follow the forward: the suite reads one interior value
  per trace (`FamilySuite.test_every_source_value_resolves_on_every_layer`,
  `FamilySuite._read_interior`, `suite.py`) because families bind them at
  different points.
- The class-scoped `model` fixture is shared by every test in the class;
  a test that needs another config loads a patched copy of the checkpoint as
  its own class (`built(patched_checkpoint, repo, ...)`, `_alibi_checkpoint()`
  in `test_falcon.py`, `_patched_checkpoint()` in `test_olmo3.py`) rather than
  mutating the fixture's; build it with `built(...)`, not at import.
- `route_delta_rule(..., "recurrent")` (`route_kernels(..., "torch")`) is
  process-wide; the hybrid tests load a second model after routing and restore
  `"chunked"` in a `finally`.
- pytest imports `tests/families/suite.py` as the bare module `suite`
  (`from suite import FamilySuite`); do not name another test helper that.
- Run pytest from the repository root: `testpaths` is relative to it.
- The fixtures load with `dispatch=True` and no `device=`, so with a GPU visible every
  tiny model lands on `cuda:0`; on a shared card a parallel run fails by the hundreds
  (CUDA out of memory, cuBLAS errors). Run with `CUDA_VISIBLE_DEVICES=""` to keep the
  suite on CPU.
- The tiny checkpoints cannot show scale-dependent behaviour: on
  `hf-tiny-v2/tiny-random-Olmo2ForCausalLM` the sublayer outputs have an RMS near 1e-6,
  far below `sqrt(rms_norm_eps)` (3e-3), so the RMSNorms act linearly and scaling a
  module's output moves the stream as scaling its contribution does. A test about norm
  scale invariance needs real weights or scaled activations.

## Related

- [contributing.md](contributing.md) — the workflow around the suite
- [transformers-compat.md](transformers-compat.md) — the tests that guard op names
- [recurrent-mixer-internals.md](recurrent-mixer-internals.md) — what the hybrid tests pin
- nnsight `docs/developing/testing.md` — the suite of the library underneath
