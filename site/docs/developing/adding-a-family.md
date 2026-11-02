# Adding a family

A family is one module under `nnterp/families/`, named after the checkpoint's `config.model_type`, with a `RENAME` dict from native names to standard names, three envoy subclasses, and an `ENVOYS` dict keying those on the family's module classes. Its test file is a `FamilySuite` subclass with a pinned tiny checkpoint. The file name is the whole registration: `nnterp.families.lookup` imports `nnterp.families.<model_type>` the first time a checkpoint of that type is loaded, and nothing else in nnterp changes. A shipped family also has a page in the [encyclopedia](https://ndif-team.github.io/nnterp/encyclopedia/), an entry under `encyclopedia/entries/` written as `encyclopedia/README.md` describes.

## The module

Llama is the family the standard vocabulary is taken from, and its module changes nothing but the container names. This is `nnterp/families/llama.py` without the names of the multimodal wrappers around a Llama text model, which the shipped file also carries:

```python
"""Llama (``LlamaForCausalLM``), the family the standard vocabulary is taken from."""

from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer, LlamaMLP

from nnterp.components import Attention, Layer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """Llama's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Llama's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """Llama's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {LlamaDecoderLayer: Layer, LlamaAttention: Attention, LlamaMLP: Mlp}
```

nnsight resolves each `RENAME` key relative to every envoy in the tree ([module access](https://nnsight.net/features/6_modules/) on nnsight.net). A key with several components, such as `"model.layers"`, resolves from the root, which is what lifts the containers out of the inner model: The alias is `model.layers`, not `model.model.layers`. A single-component key binds on every envoy with a child of that name, so GPT-2's `"ln_1": "input_layernorm"`, `"attn": "self_attn"` and `"ln_2": "post_attention_layernorm"` rename every block at once. A key that resolves nowhere is skipped.

The three classes are defined even where nothing changes, because the suite asserts that the block envoys are exactly the family's `Layer`, `Attention` and `Mlp`, and each docstring records why the base holds. A block that returns `(hidden_states, ...)` rather than the tensor, as GPT-J's, BLOOM's, MPT's and Falcon's do, sets `returns_tuple = True` on its `Layer`. `layer_output` unwraps the tuple either way, and the flag is what `skip_layers` needs to hand back the right shape.

## Checking the module before it ships

Until the file is in the package, pass the module to one load as `family=` ([Registering](registering.md)). SmolLM2 is a Llama checkpoint (`model_type` `llama`). Loaded eager, nothing on it is unavailable, and the contributions add up to the block's output:

```python
import torch
from nnterp import StandardizedTransformer

import llama  # the module above, saved as llama.py beside this script

model = StandardizedTransformer("HuggingFaceTB/SmolLM2-135M", family=llama, attn_implementation="eager")
print(model.family is llama)
# True
print({name: reason for name, reason in model.support().items() if reason})
# {}
with model.trace("The Eiffel Tower is in"):
    block = model.layers[5]
    x = block.layer_input.save()
    attn = block.self_attn.attention_output.save()
    mlp = block.mlp.mlp_output.save()
    out = block.layer_output.save()
print(torch.equal(x + attn + mlp, out))
# True
```

## Redefining a value

Because Gemma-2's block norms each sublayer's output before adding it, what the attention adds to the stream is the post-norm's output, not the module's. Its `Attention` redefines `attention_output` on that sibling norm, keeping the name, the `-> Residual` annotation and the description, and changing only where the value is read:

```python
from nnterp.components import Attention, EProperty, Residual


class Attention(Attention):
    @EProperty("../post_attention_layernorm.output", description="What the attention adds to the residual stream: the post-attention norm's output")
    def attention_output(self, value) -> Residual:
        return value
```

[Overriding a value](overriding-values.md) has every form the path takes, and the markers for a value a family does not have.

## A size the config spells its own way

The root's sizes (`num_layers`, `hidden_size`, `vocab_size`, `num_heads`, `num_kv_heads`, `head_dim`, `qk_head_dim` and `intermediate_size`) are read off the config by Llama's rule, which for `intermediate_size` is `config.intermediate_size`. GPT-2's config has no such key. Its module defines a function of the same name taking the model, and the root calls it in place of the rule:

```python
# nnterp/families/gpt2.py
def intermediate_size(model: "StandardizedTransformer") -> int:
    """The MLP width is ``n_inner``, ``None`` meaning four times the hidden size; the config's ``intermediate_size`` is never read by the model."""
    return model.config.n_inner or 4 * model.hidden_size
```

```python
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2")
print(model.config.n_inner, model.intermediate_size, model.layers[0].mlp.intermediate_size)
# None 3072 3072
```

A block's own `mlp.intermediate_size` is read off the module rather than the config. Falcon's module spells `num_kv_heads` the same way. A model that scales or caps its logits after `lm_head` defines `def project_on_vocab(model, hidden)` in the module (Cohere multiplies by `logit_scale`), which is bound in place of the root's, so the logit lens on the last block still equals `logits`.

## The test file

One file per family under `tests/families/`, subclassing `FamilySuite` from `tests/families/suite.py`. Llama's is the minimum:

```python
"""Llama, end to end: the family the vocabulary is taken from."""

from suite import FamilySuite, LLAMA_ROWS

from nnterp.families import llama


class TestLlama(FamilySuite):
    REPO = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    FAMILY = llama
    NATIVE = LLAMA_ROWS
```

`NATIVE` maps standard paths to native paths, and `rows()` builds it from the container and the native names: `LLAMA_ROWS` is `rows("model", "layers", "embed_tokens", "norm")`, and GPT-2's is `rows("transformer", "h", "wte", "ln_f", attn="attn", ln1="ln_1", ln2="ln_2")`. A quirk of the checkpoint is a class attribute:

| attribute | meaning |
| --- | --- |
| `EXPECTED_UNAVAILABLE` | `support()` key to a substring of the reason, for the values this checkpoint lacks; every other value must report `None` |
| `REFUSES_IN_PLACE_QKV` | torch refuses in-place edits on queries that are views out of one `split` (GPT-2, GPT-BigCode, MPT) |
| `ATTENTION_NORM`, `MLP_NORM` | the block's own module whose output enters each sublayer, where it is not `input_layernorm` and `post_attention_layernorm` (Gemma-2's `MLP_NORM` is `pre_feedforward_layernorm`) |
| `LOAD_KWARGS` | extra load arguments the checkpoint needs |

Run it:

```
pytest tests/families/test_llama.py -q
# 41 passed, 12 skipped, 2 warnings in 12.62s
```

The suite loads the checkpoint eager and asserts, among other things, that:

- every alias reaches its native module, and the block envoys are the family's classes
- `support()` matches what reads and what raises
- `layer_input + attention_output + mlp_output == layer_output` on every block
- the standardized model equals a raw `TransformersModel`
- every source-located value resolves on every attention block
- each value's tensor matches its layout, and the sizes match the weights

[Testing](testing.md) lists all of it. The skipped tests are the mixture-of-experts ones, which a dense model has none of. A failure in `test_every_source_value_resolves_on_every_layer` names the operation the forward does not have, and [Finding source ops](finding-source-ops.md) is how to find the one it does.
