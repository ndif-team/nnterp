# Testing

`HF_HUB_OFFLINE=1 python -m pytest` from the repository root runs every family's file under `tests/families/` against its pinned tiny checkpoint, and the root tests under `tests/`. `FamilySuite` in `tests/families/suite.py` is the contract: 53 methods every family inherits, each an end-to-end statement about a standardized model. A family's file names its checkpoint, its native paths and its quirks as class attributes and adds only what is specific to it.

```bash
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 python -m pytest tests/families/test_gpt2.py -q -p no:cacheprovider
# 42 passed, 12 skipped, 2 warnings in 5.72s
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 python -m pytest tests/test_registry.py -q -p no:cacheprovider
# 23 passed, 7 warnings in 14.24s
```

The checkpoints are cached, and `HF_HUB_OFFLINE=1` keeps a run from touching the Hub. The fixtures load with `dispatch=True` and no `device=`, which puts every model on a visible GPU, and `CUDA_VISIBLE_DEVICES=""` keeps a run on the CPU. The twelve skips on GPT-2 are the mixture-of-experts methods, which skip on a dense model. `pyproject.toml` sets `testpaths = ["tests"]` and `addopts = "--dist loadgroup"`, so a `pytest -n` run with xdist keeps a class marked `xdist_group` on one worker. `tests/conftest.py` imports nnsight before anything else, since a transformers modeling module imported first can crash the interpreter on import, and pins one torch thread per xdist worker. pytest imports `tests/families/suite.py` as the bare module `suite`, and no other test helper may take that name.

## What `FamilySuite` asserts

Two class-scoped fixtures load the checkpoint once per class: `model`, a `StandardizedTransformer` with `dispatch=True, attn_implementation="eager"` plus `LOAD_KWARGS`, and `raw_model`, the same checkpoint as a plain `TransformersModel`. The methods, by what they pin:

| group | methods | assert |
|---|---|---|
| names | `test_family_resolved`, `test_standard_names_alias_native_envoys`, `test_no_inner_model_alias`, `test_every_layer_is_renamed`, `test_envoy_classes`, `test_trace_through_standard_names` | `model.family is FAMILY`; every `NATIVE` row resolves to the native envoy itself; nothing is bound at `model.model`; every block carries the block rows; every block, attention, MLP and mixer is exactly the family's class and subclasses the component base; a trace through the standard names gives `hidden_size` and `vocab_size` widths |
| availability | `test_support_lists_every_standard_value`, `test_support_is_what_this_family_expects`, `test_support_matches_what_reads` | `support()` has exactly the expected keys; each `EXPECTED_UNAVAILABLE` key carries its reason substring on every block and every other key is `None`; on block 0 an available name is an `EProperty` and an unavailable one raises `Unavailable` with that reason |
| boundaries | `test_layer_output_is_the_tensor`, `test_layer_input_is_the_block_input`, `test_every_value_reads_a_tensor_on_every_layer`, `test_contribution_identity`, `test_sublayer_inputs_are_the_normed_stream`, `test_module_boundaries`, `test_sublayer_interiors`, `test_renamed_model_equals_raw_model`, `test_boundary_writes_land` | `layer_output` equals `.output` or its first element; `layer_input + attention_output + mlp_output == layer_output` on every block at 8 ulp of the block's dtype; what enters each sublayer is `ATTENTION_NORM`'s and `MLP_NORM`'s output; `layer_mid`, the norm outputs, `attention_premix`, `mlp_activation` and `mlp_neurons` are the module boundaries they name and a write to each moves the logits; the renamed model equals the raw one; assigning and zeroing the three boundary values moves the logits |
| the pattern | `test_probabilities_are_a_pattern`, `test_attention_interior_under_generate`, `test_pattern_across_layers_and_traces`, `test_written_pattern_moves_the_logits`, `test_every_source_value_resolves_on_every_layer` | `[batch, num_heads, seq, seq]`, the model dtype, rows summing to one (or below one with `ATTENTION_SINK`), lower-triangular; a decode step's pattern has one query row; two traces agree; a written pattern and a zeroed head move the logits; every value with `inside_forward()` reads on every attention block |
| the interior | `test_interior_shapes`, `test_interior_writes_are_causal`, `test_interior_in_place_edits` | the queries, keys, values, scores and head outputs have their layouts and `softmax(scores)` is the pattern (`pattern_from_scores` is overridable); zeroing each moves the logits and zeroed head outputs leave a position-independent contribution; in-place edits land, or queries raise `RuntimeError` mentioning `view` with `REFUSES_IN_PLACE_QKV` |
| methods | `test_skip_layers_hands_the_stream_straight_through`, `test_skip_layers_with_a_given_stream`, `test_steer`, `test_project_on_vocab_is_the_logit_lens` | skipping blocks 1 to the last leaves block 0's stream as the final one; `skip_with` zeros reach the next block; `steer` moves the last position by `2 * vector` and no other; `project_on_vocab` on the last block's output is `logits` |
| layouts | `test_values_match_their_annotations` | every value with a `layout` on the root, a block, its attention, its MLP, its mixture and its mixer is an instance of the alias and each named axis matches `axis_sizes` |
| the input | `test_input_accessors`, `test_attention_mask_under_generate`, `test_assigning_input_ids_runs_other_ids` | `input_ids`, `input_size` and `attention_mask` are `[1, n]`; a decode step's mask covers the cache; assigning another prompt's ids reproduces its logits and assigning `input_size` raises |
| the root | `test_logits_are_the_models_output`, `test_assigning_logits_replaces_the_result`, `test_token_embeddings_are_the_embedding_output`, `test_next_token_probs`, `test_sizes_match_the_model`, `test_per_module_sizes_match_each_block`, `test_repr_lists_the_values` | `logits` is `output.logits`; assigning it zeros `tracer.result.logits`; `next_token_probs` is the last position's softmax and refuses assignment; the root sizes match the pattern, the stream and the projections, and each block's own sizes match its tensors; the block repr lists the values |
| a mixture | the twelve `test_moe_*`, `test_expert_*`, `test_routed_*`, `test_writes_to_the_shared_*`, `test_zeroing_*`, `test_ablating_*`, `test_rerouting_*`, `test_writing_router_logits_*` methods | on the first block with an `mlp` that is a `Moe` running a mixture: the routing pair's shapes, `expert_outputs` summing to `routed_output`, `routed_output + shared_expert_output` being the module's output, a zeroed slot and an ablated expert changing exactly the tokens they touch, a rerouted slot matching a hand computation, written router logits deciding the routing, and per-invoke rows; skipped on a dense model |

Reads in one trace follow the forward, and families bind the interior values at different points, so the interior and layout tests read one value per trace. Containers filled inside a trace block are bound outside it, since a name first bound in the block does not survive it.

## How a family file adds specifics

Most files are the class attributes alone. `tests/families/test_gpt2.py` sets `REPO`, `FAMILY`, `NATIVE = rows("transformer", "h", "wte", "ln_f", attn="attn", ln1="ln_1", ln2="ln_2")` and `REFUSES_IN_PLACE_QKV = True`, and adds one test that flipping `reorder_and_upcast_attn` on the config makes the interface unavailable. `rows(container, layers, embed, norm, ...)` builds the standard-to-native map, and `LLAMA_ROWS` is the Llama layout.

| attribute | states |
|---|---|
| `EXPECTED_UNAVAILABLE` | `support()` key to a substring of the reason, for the values this checkpoint lacks; a module no block has is absent from `support()` and is not listed here |
| `ATTENTION_NORM`, `MLP_NORM`, `MLP_NORM_BEFORE_ATTENTION` | which block module's output enters each sublayer (`None` when the block input does), and whether the MLP's norm fires first |
| `REFUSES_IN_PLACE_QKV`, `ATTENTION_SINK`, `KV_HEADS_EXPANDED`, `QUERY_GATED` | queries that are split views, a sink column, keys and values already expanded to `num_heads`, a `q_proj` twice as wide |
| `MLP_WIDTH_KEY`, `LOAD_KWARGS` | a config key naming the MLP width when it is not `intermediate_size`; extra load arguments (DBRX needs `dtype=torch.float32`) |
| `MOE_UNAVAILABLE`, `ROUTER_EXTRA_CLASSES` | mixture values the mixture lacks; router classes beyond the experts |
| `DECODE_MASK_COVERS_CACHE`, `MID_IDENTITY`, `PREMIX_IS_HEAD_OUTPUTS`, `NEURONS_FROM_ACTIVATION` | a pure state-space model's decode mask, a gated stream, a norm between the heads and the output projection, a fused `gate_up_proj` |

A family with a checkpoint that needs another config or another kernel path makes a second class rather than mutating the shared fixture. `test_falcon.py` has three classes for the 7B layout, the 7B layout with alibi written into a copy of `config.json`, and the 40B layout, and `test_olmo3.py` symlinks the snapshot into a temporary directory and rewrites only `config.json`. The hybrid files override `test_every_layer_is_renamed` for the per-block `self_attn` and `linear_attn` split. The Mamba-1 files mix `SelectiveScanSuite` (`tests/families/scan_suite.py`) in before `FamilySuite`: A class-scoped autouse fixture routes the family to the pure-torch kernels for the class and restores the default afterwards, which matters because the routing is process-wide. The Mamba-2 files take their mixer tests from `tests/families/ssd.py`. A multimodal wrapper runs the whole `FamilySuite` loaded with `LOAD_KWARGS = {"task": "image-text-to-text"}`, and its vision side runs `VisionSuite` from `tests/families/vision_suite.py`: the tower names, the tower's identity, `layers[0].input[vision.image_token_mask] == vision.image_features`, causal edits, and no `vision` host on a text-only checkpoint or load. `tests/families/test_default.py` runs the default family on several shipped families' checkpoints with the contributions listed unavailable, and checks that every value it serves is bit-identical to the dedicated family's.

## The root tests

```bash
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 python -m pytest tests/test_base.py tests/test_prompt_utils.py tests/test_nnsight_utils.py tests/test_order.py -q -p no:cacheprovider
# 51 passed in 26.67s
```

| file | covers |
|---|---|
| `tests/test_registry.py` | every module under `nnterp/families/` is named after its type; `import nnterp` pulls in no `transformers.models.*.modeling_*` module and `lookup("gpt2")` imports only that family, in a subprocess; an unknown `model_type` falls back to the default family with a warning; `register` adds and overrides; a preloaded module uses its own config; a caller's `rename` and `envoys` merge over the family's; the remote key is `TransformersModel`; a size or `project_on_vocab` defined on a family wins over the root's; `family=` takes a module or a `SimpleNamespace` for one load; a size refuses assignment |
| `tests/test_base.py` | a tuple-returning block is unwrapped and rewrapped; an `unavailable(...)` marker is listed, raises on read and makes `hasattr` raise; paths with `../`, `/` and `select` functions resolve and write; `route_kernels` round-trips a module's bindings; `per_call` separates invokes and replays an edit; `axis_size` resolves by name up the tree; `probs_to_dict` keeps tokens that decode alike |
| `tests/test_prompt_utils.py`, `tests/test_nnsight_utils.py` | the helpers, on GPT-2 with left padding |
| `tests/test_order.py` | the measured forward order and its ranks |
| `tests/test_encyclopedia.py` | every encyclopedia entry builds from its pinned checkpoint |
| `tests/vllm_families/` | the vLLM engine against the transformers one; collected only where vLLM imports and a GPU is visible |

The procedure for a transformers release that moves an operation name is on [transformers compatibility](transformers-compatibility.md).
