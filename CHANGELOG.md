# Changelog

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
