# Changelog

## Unreleased

### New Features

- **Whole-model places are accessors too.** `embeddings_input` (the token
  ids), `embeddings_output`, `ln_final_output` and `lm_head_output` are rows of
  the same address table as the per-layer accessors (`Address(per_layer=False)`),
  so they carry a `select`/`Lens`, an availability reason and a place in forward
  order like every other. Each is a property on the model — `model.lm_head_output`
  is the tensor inside a trace, `model.lm_head_output = value` writes it — and
  `model.internals["lm_head_output"]` is the accessor behind it; indexing one by
  a layer is refused by name. `model.lm_head.output` keeps working for the raw
  module. `lm_head_output` is the head module's output, uncapped on Gemma-2;
  `model.logits` remains the capped logits the model predicts from.
- **Availability per layer.** `Address.unavailable` may be a function of the
  layer's module, and `model.internals.status(layer=i)` answers for one layer:
  on DeepSeek, `mlps_activation` is available on the dense first blocks and says
  "a mixture-of-experts layer" on the rest, before any trace. `status()` with no
  layer is the model-wide answer. `internals.read(*names, layer=i)` reads
  per-layer and whole-model places together in forward order
  (`internals.rank(name, layer)`); `read(layer, *names)` still works.
- **Six accessors inside the block**, each defined by what it is *of* the block
  rather than by a module name: `layers_mid` (the residual stream after
  attention), `attentions_norm_output` and `mlps_norm_output` (what each sublayer
  consumes), `attentions_premix` (the output projection's input: every head's
  result side by side, `num_heads * head_dim` wide, which is not `hidden_size` on
  Qwen3 or Gemma), `mlps_activation` and `mlps_neurons` (the down projection's
  input; equal to the activation only on an ungated MLP). The children they live
  on are spelled differently per family (`o_proj` / `c_proj` / `dense` /
  `out_proj`, ...); the name is read off the model's module tree, and a family
  or a `RenameConfig(addresses=...)` row overrides it. On every family that has
  them, `layers_mid == layers_input + attentions_output`, `layers_output ==
  layers_mid + mlps_output` and `mlps_norm_output == mlps_input`.
- **`model.block_structure`**: `"pre_norm"`, `"sandwich_norm"` (Gemma-2/3),
  `"post_norm"` (OLMo-2), `"parallel"` (GPT-NeoX, GPT-J, Falcon, Phi, StableLM-2,
  read from the config's flags) or `"residual_inside"` (BLOOM, MPT, DBRX). It
  decides which accessors exist: a parallel block has no `layers_mid` and no
  pre-MLP norm, and a post-norm block no pre-sublayer norms. What a model lacks
  is in `model.internals.status()` before any trace, with the reason; a mixture
  of experts has no `mlps_activation` / `mlps_neurons`, per layer where a model
  mixes dense and sparse blocks (DeepSeek), and OPT's `mlps*` accessors now say
  that it has no MLP module instead of failing inside a trace.
- **`model.head_dim`, `model.qk_head_dim`, `model.num_kv_heads`,
  `model.intermediate_size`.** `head_dim` is the config's own where it states one
  (it is not `hidden_size // num_heads` on Qwen3 or Gemma) and the value-side
  width under multi-head latent attention (DeepSeek), where `qk_head_dim` differs.
  `intermediate_size` follows each family's spelling (`n_inner` or four times
  hidden on GPT-2, `ffn_hidden_size`, `ffn_dim`). All four are checked against
  the tensors and modules of 24 families in `tests/test_block_invariants.py`.

- **Hybrid linear/softmax attention models (Qwen3-Next, Qwen3.5, Qwen3.6).** A
  Gated DeltaNet mixer keeps its `linear_attn` name instead of being renamed to
  `self_attn`, so every block exposes exactly one of `layers[i].self_attn` /
  `layers[i].linear_attn`. `model.attention_layers` and
  `model.linear_attention_layers` list the two kinds of blocks, computed from the
  block structure at load and cross-checked against the config's `layer_types`.
  `attentions[i]`, `attentions_input[i]`, `attentions_output[i]` and
  `attention_probabilities[i]` raise a `RenamingError` on a linear-attention
  layer; the renaming checks, the attention-probability validation and
  `attention_probabilities.print_source()` use the first softmax-attention layer.

### Changes

- **`attentions_output` / `mlps_output` are the contributions on Gemma-2/3 and
  OLMo-2 (behaviour change).** These families normalize a sublayer's output
  before adding it to the residual stream, so the tensor added is
  `post_attention_layernorm` / `post_feedforward_layernorm`'s output, and the two
  accessors now target those. `layers_input[i] + attentions_output[i] +
  mlps_output[i] == layers_output[i]` holds on them as it does everywhere else;
  before, the accessors returned the pre-norm module outputs (on
  google/gemma-2-2b the attention output has a norm 17x smaller than the stream
  it was taken to be part of). Code that wants the raw module output reads
  `attentions[i].output` / `mlps[i].output`.

- **Per-layer tuple detection in the accessors.** `layers_output[i]` and the
  other I/O accessors unwrap a tuple per access instead of assuming every layer
  returns the same structure, so layers can be accessed in any order.
  `LayerAccessor.returns_tuple(layer)` replaces the `returns_tuple` property; the
  renaming checks read every layer output in forward order, `skip_layers` uses
  the per-layer record, and `detect_layer_output_type()` records the layers not
  accessed yet.

- **`remote=True` keeps the checkpoint off the client.** It sets
  `allow_dispatch=False`, so every load-time check runs with `scan()` on the meta
  model, and no request is sent to NDIF during construction: after
  `StandardizedTransformer(name, remote=True, enable_attention_probs=True)`,
  `model.dispatched` is `False` and the parameters are on the `meta` device.
  `check_attn_probs_with_trace` defaults to `None`, meaning `True` for a local
  model and `False` for a remote one (a shape check under `scan()`); passing
  `check_attn_probs_with_trace=True` with `remote=True` runs the full check as
  traces on NDIF.

### Fixes

- **`attn_implementation="eager"` is accepted with `enable_attention_probs=True`.**
  `StandardizedTransformer(name, enable_attention_probs=True,
  attn_implementation="eager")` loads, and the keyword reaches the model once. A
  non-eager value raises the `ValueError` naming the conflict.

- **Attention probabilities work again on transformers >= 5.** GPT-2's
  `eager_attention_forward` changed from `module.attn_dropout(attn_weights)` to
  `nn.functional.dropout(...)` in transformers 5, which renames the nnsight source
  operation from `module_attn_dropout_0` to `nn_functional_dropout_0`. Loading
  `StandardizedTransformer("openai-community/gpt2", enable_attention_probs=True)`
  failed at construction with `AttributeError: ... has no operation
  'module_attn_dropout_0'`. The attention-probability accessors now try the known
  spellings in order via `first_available_op`, so one nnterp works across
  transformers 4.x and 5.x. Verified on GPT-2, Bloom, Llama-style (SmolLM2) and
  GPT-NeoX (Pythia).

- **Attention probabilities work on nnsight 0.8.** nnsight 0.8 labels an
  assignment and a call on one per-name counter, so the attention forward's
  `attention_interface = ...` binding is `attention_interface_0` and the call is
  `attention_interface_1`. The source functions read the call. Requires
  `nnsight>=0.8`, now declared.

## v1.3.0

### Breaking Changes

- **Python 3.10+ required** — dropped Python 3.9 support.
- **nnsight >=0.6 required** — bumped minimum dependency from 0.5 to 0.6.
- **Replaced loguru with standard library logging** — all logging now uses `logging.getLogger("nnterp")`. Users who configured loguru sinks for nnterp should switch to `logging.getLogger("nnterp").setLevel(...)` etc.

### New Features

- **`token_positions` and `batch_index` parameters for `steer()`** — fine-grained control over which tokens and batch elements are steered. Both can be combined:
  ```python
  with model.trace(["prompt A", "prompt B"]):
      model.steer(layers=1, steering_vector=v, batch_index=0, token_positions=[0, 1])
  ```
  The old `positions` parameter is deprecated (still works, emits `DeprecationWarning`).

- **`remote` parameter on `StandardizedTransformer`** — `remote=True` automatically sets `allow_dispatch=False` and registers nnterp for NDIF remote execution.

- **Multimodal model detection** — `check_model_renaming()` now detects heterogeneous layer types (e.g. self-attention + cross-attention in vision-language models) and raises an informative error. Bypass with `allow_multimodal=True`.

### Internal

- Removed broken `steer` import from `nnterp.interventions` in tests/demos (was already a dead reference).
- Added comprehensive steer tests covering `token_positions`, `batch_index`, and their combination.
- Updated README with new branding, corrected documentation URLs, and new steer API examples.
