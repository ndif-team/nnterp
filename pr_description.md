## Summary

Adds `StandardizedVLM` for vision-language models and `StandardizedVLLM` for the experimental vLLM backend, both sharing a new `StandardizationMixin` that extracts the common standardized interface. Also extends attention probability support to 4 new architectures and improves robustness of model renaming.

## Features

- **`StandardizedVLM`**: New wrapper extending nnsight's `VisionLanguageModel` with the same standardized accessors as `StandardizedTransformer` (layers, attention, MLP, logits, etc.). Supports image inputs via `model.trace(prompt, images=...)`.
- **`StandardizedVLLM`**: New experimental wrapper for the vLLM backend (`standardized_vllm.py`), gated behind `allow_experimental_vllm=True`. Handles vLLM-specific quirks: no inplace ops for steering, meta-device weights for `project_on_vocab`, prefix caching safety, and HF→vLLM kwarg translation.
- **`load_model()` entrypoint**: Auto-detects VLMs via `detect_automodel()` and returns the appropriate wrapper. Supports `use_vllm=True` for vLLM backend.
- **`detect_automodel()`**: Inspects model config to pick the right `AutoModel` class (priority: `AutoModelForImageTextToText` > `AutoModelForCausalLM` > `AutoModelForSeq2SeqLM`).
- **`StandardizationMixin`**: Refactored shared logic out of `StandardizedTransformer` into a mixin, used by all three wrappers.
- **vLLM `.generate()` / `.trace()` split**: `.trace()` defaults to `max_tokens=1` (single forward pass); `.generate()` enables multi-token autoregressive generation with `tracer.all()` support.
- **Attention probability support** for Qwen2Moe, Dbrx, StableLm, and GptOss (with sink token handling for GptOss where probs don't sum to 1).
- **Heterogeneous submodule warnings**: Detects mixed attention/MLP types across layers (e.g. dense + MoE) and warns.
- **Llama-4 support**: Added `feed_forward` to `MLP_NAMES`.
- **vLLM documentation page** (`docs/vllm.rst`): Full guide covering setup, loading, interventions, generation, differences from HF backend, and known limitations.

## Fixes

- **Steer dtype casting**: `steer()` now casts the steering vector to match the layer output dtype, preventing dtype mismatch errors.
- **`check_io` now works for vLLM**: Refactored to handle vLLM's flat `(seq_len,)` input shape (no batch dim) and skip `lm_head.output` check (vLLM computes logits in a separate phase). Extracted `_check_tensor()` helper to deduplicate validation.
- **`input_size` works for vLLM**: Returns `(seq_len,)` instead of raising `NotImplementedError`.
- **bfloat16 attention tolerance**: Raised `atol` from `1e-5` to `1e-2` for bfloat16 attention probability sum checks.
- **`dummy_inputs()` now includes `attention_mask`**, fixing scan/trace failures for models that require it.
- **`final_layernorm` added to `LN_NAMES`** for models using that naming convention.
- **`model.lm_head` added to `LM_HEAD_NAMES`** (with model prefix expansion).
- **`try_with_scan` respects `model.remote`** for NDIF remote execution.
- **`hf_kwargs_to_vllm_kwargs` cleanup**: Fixed `max_new_tokens` → `max_tokens` mapping (was incorrectly `max_num_tokens`), added docstring, simplified logic.
- **`tracer.stop()` temporarily disabled** in `try_with_scan` (pending upstream nnsight fix).

## Tests

- **`test_vllm.py`** (new, 21 tests): Per-test vLLM engine isolation via function-scoped fixture with proper engine shutdown. Covers: experimental flag gating, prefix caching safety, `check_renaming=True` validation, layer activations, logits, skip layer, steering, multi-token generation (`tracer.all()`), single-token trace default, `project_on_vocab` guard, unsupported feature guards, and `load_model` integration.
- **`test_vlm.py`** (new): Tests for `load_model` autodetection, VLM properties, layer activations, `project_on_vocab`, skip layer, steering (including batch-indexed steering). VLM test models are auto-discovered from HuggingFace toy model configs.
- **Test config updates**: Added skip patterns with comments for gemma-3n, llama-3.2-vision, falcon-h1, jamba.

## Docs

- **`docs/vllm.rst`** (new): Full vLLM backend documentation covering setup, model loading, standardized interface, interventions, generation, HF/vLLM differences, unsupported features, and known limitations.
- Updated `basic-usage.rst` with VLM loading examples and known limitations.
- Updated `changelog.rst` with all additions and changes.
- Updated `README.md` to use `load_model()` in the quickstart example.
- Removed "Add vLLM support" from the roadmap.

## Known limitations

- **vLLM attention probabilities not supported yet** (raises `NotImplementedError`).
- **vLLM `attention_mask`** is not in vLLM's inputs dict (only `input_ids`, `positions`, `intermediate_tensors`, `inputs_embeds` are available).
- **vLLM `add_prefix_false_tokenizer`** not supported.
- **`tracer.stop()` disabled**: Temporarily commented out in `try_with_scan` pending upstream nnsight fix.
- **Mllama/Llama-3.2-Vision**: Cross-attention layers only fire with image inputs; skipped in tests.
- **Gemma 3n**: AltUp 4D hidden states not supported (see #35).
