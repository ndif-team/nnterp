---
name: Find Attention Probability Hooks
description: Discover nnsight hookpoints for transformer model attention probabilities. Use when adding attention probability support to a new model architecture in nnterp, or when attention probability access fails for a model.
---

# Find Attention Probability Hooks

A systematic workflow for discovering the correct nnsight hookpoints to access attention probabilities in transformer models.

## Quick Start

Use this workflow when adding attention probability support for a new model architecture:

```python
import torch as th
from nnterp import StandardizedTransformer

# 1. Load model WITHOUT enabling attention probs
model = StandardizedTransformer(
    "model-name",
    enable_attention_probs=False,
    device_map="cpu",
    attn_implementation="eager"
)

# 2. Use .trace() and .source to find hookpoints
with model.trace(th.tensor([[1, 2, 3]])):
    attn = model.attentions[0]

    # Check for sub-modules (e.g., .attn in Dbrx)
    if hasattr(attn, 'attn'):
        attn = attn.attn

    # Print source to see available hookpoints
    print(attn.source)
```

## Core Principle

**ALWAYS use `.trace()` + `.source`, NEVER read transformer files directly**

Why? The `.source` attribute shows actual nnsight hookpoints (e.g., `nn_functional_dropout_0`). Reading Python files only shows code, not hookpoint names.

## Hookpoint Priority

Look for these hookpoints in order:

1. **`attention_interface_0`** - Standard pattern (LLaMA-like models)
   - Check `.source.nn_functional_dropout_0` inside it

2. **`nn_functional_dropout_0`** - Direct functional dropout
   - Examples: Qwen2Moe, Dbrx

3. **`self_attn_dropout_0` or `self_attention_dropout_0`** - Module dropout
   - Examples: StableLm, Bloom

4. **`F_softmax_0`** - Special cases (models with sink tokens)
   - Example: GptOss

## Implementation Steps

### 1. Find the Hookpoint

```python
with model.trace(th.tensor([[1, 2, 3]])):
    attn = model.attentions[0]

    # Check for nested structures
    if hasattr(attn, 'attn'):
        print("Using .attn sub-module")
        attn = attn.attn

    print("SOURCE:")
    print(attn.source)

    # Look for dropout or softmax operations
    ops = [a for a in dir(attn.source) if 'dropout' in a.lower() or 'softmax' in a.lower()]
    print(f"Relevant ops: {ops}")
```

### 2. Test the Hookpoint

```python
with model.trace(th.tensor([[1, 2, 3]])):
    # Test accessing the output
    probs = model.attentions[0].source.HOOKPOINT_NAME.output.save()

print(f"Shape: {probs.shape}")
print(f"Sum of last dim: {probs.sum(dim=-1).mean():.4f}")  # Should be ~1.0
```

### 3. Implement the Function

Add to `nnterp/rename_utils.py`:

```python
def modelname_attention_prob_source(attention_module, return_module_source: bool = False):
    if return_module_source:
        return attention_module.source  # or .attn.source if nested
    else:
        return attention_module.source.HOOKPOINT_NAME
```

### 4. Register the Model

**In `nnterp/utils.py`:**
```python
try:
    from transformers import YourModelForCausalLM
except ImportError:
    YourModelForCausalLM = ArchitectureNotFound
```

**In `nnterp/rename_utils.py` imports:**
```python
from .utils import (
    # ... existing imports ...
    YourModelForCausalLM,
)
```

**In `AttentionProbabilitiesAccessor.__init__()`:**
```python
elif isinstance(model._model, YourModelForCausalLM):
    self.source_attr = yourmodel_attention_prob_source
    # Add this if model has non-standard attention shape (e.g., sink tokens):
    # self.has_non_standard_shape = True
```

## Pattern Examples

### Direct Dropout (Qwen2Moe)
```python
def qwen2moe_attention_prob_source(attention_module, return_module_source: bool = False):
    if return_module_source:
        return attention_module.source
    else:
        return attention_module.source.nn_functional_dropout_0
```

### Nested Module (Dbrx)
```python
def dbrx_attention_prob_source(attention_module, return_module_source: bool = False):
    if return_module_source:
        return attention_module.attn.source
    else:
        return attention_module.attn.source.nn_functional_dropout_0
```

### Module Dropout (StableLm)
```python
def stablelm_attention_prob_source(attention_module, return_module_source: bool = False):
    if return_module_source:
        return attention_module.source
    else:
        return attention_module.source.self_attention_dropout_0
```

## Quick Reference Table

| Model | Hookpoint | Notes |
|-------|-----------|-------|
| Qwen2Moe | `nn_functional_dropout_0` | Direct |
| Dbrx | `.attn.source.nn_functional_dropout_0` | Nested |
| StableLm | `self_attention_dropout_0` | Module |
| LLaMA-like | `attention_interface_0.source.nn_functional_dropout_0` | Default |
| Bloom | `self_attention_dropout_0` | Module |
| GPT-2 | `attention_interface_0.source.module_attn_dropout_0` | Module via interface |
| GPT-J | `self__attn_0.source.self_attn_dropout_0` | Nested |

## Testing Checklist

- [ ] Model loads with `enable_attention_probs=False`
- [ ] Can access `model.attentions[0].source` in trace
- [ ] Identified correct hookpoint name
- [ ] Probabilities have expected shape
- [ ] Probabilities sum to ~1.0 (or documented exception)
- [ ] Changing probabilities changes model output
- [ ] Function added to `rename_utils.py`
- [ ] Model class imported in `utils.py`
- [ ] Registered in `AttentionProbabilitiesAccessor.__init__()`

## Related Files

- `nnterp/rename_utils.py` - Attention probability source functions
- `nnterp/utils.py` - Model class imports
- `nnterp/standardized_transformer.py` - StandardizedTransformer class
