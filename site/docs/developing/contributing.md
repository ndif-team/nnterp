# Contributing

A change is a branch off `main`, the family's test file run before and after, the whole suite before the pull request, and the docs page the change touches in the same pull request. Code follows nnsight's `STYLE.md` with the additions below, and every value, family and helper is written so that `support()`, the repr and the suite describe it without anyone opening the source. The example is a family for one load, written the way a shipped module is.

```python
from types import SimpleNamespace

import torch
import nnterp
from nnterp import StandardizedTransformer
from nnterp.components import Attention, EProperty, Layer, Mlp, Residual
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2MLP

RENAME = {"transformer.wte": "embed_tokens", "transformer.h": "layers", "transformer.ln_f": "norm", "ln_1": "input_layernorm", "attn": "self_attn", "ln_2": "post_attention_layernorm"}


class Layer(Layer):
    """GPT-2's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """GPT-2's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """GPT-2's MLP, with one value of its own."""

    @EProperty("c_fc.output", description="The up projection's output, before the activation")
    def mlp_preactivation(self, value: torch.Tensor) -> Residual:
        return value


def intermediate_size(model):
    """The MLP width is ``n_inner``, ``None`` meaning four times the hidden size."""
    return model.config.n_inner or 4 * model.hidden_size


family = SimpleNamespace(RENAME=RENAME, ENVOYS={GPT2Block: Layer, GPT2Attention: Attention, GPT2MLP: Mlp}, intermediate_size=intermediate_size)
model = StandardizedTransformer("openai-community/gpt2", family=family, dispatch=True, attn_implementation="eager", device="cpu")
print(model.family is family, model.intermediate_size)                                   # True 3072
print(model.support()["mlp.mlp_preactivation"], str(type(model.layers[0].mlp).mlp_preactivation))
# None (mlp_preactivation) -> Residual [batch seq hidden_size]: The up projection's output, before the activation
with model.trace("The Eiffel Tower is in"):
    pre = model.layers[0].mlp.mlp_preactivation.save()
print(tuple(pre.shape))   # (1, 7, 3072)
```

A shipped family is the same content as a module under `nnterp/families/` named after the `model_type`, with the transformers import at its top and a module docstring that teaches the family ([Adding a family](adding-a-family.md)). The annotation `-> Residual` is what gives the value its layout and its repr line.

## House style

- Present tense, current behavior. The source and the docs say what is: no "used to", no issue numbers, no "fixed", no TODO in prose. Where code exists because something once broke, write the constraint and its reason, not the incident.
- A docstring states the contract; a comment says why. A module docstring teaches the family or the concept, a value's docstring says what the tensor is, its layout and how writes behave, and an inline comment names the failure averted or the alternative rejected. A comment that paraphrases the line under it is deleted.
- One word, one meaning: family, name, value, contribution, available and unavailable, interface, hybrid, as the [Glossary](../reference/glossary.md) fixes them.
- A description on a value is a capitalized sentence without a trailing period that names what the tensor is, which is what `repr(model)` and `support()` show.
- One family per module, named after its `model_type`: The file name is the registry entry. Families import their modeling module at the top of the family module only, never from `nnterp/components/` or `nnterp/standardized.py`, so `import nnterp` loads no modeling code.
- Every value is annotated with a layout name (`-> Residual`, `-> Pattern`, `-> Keys`), never an inline `Float[Tensor, "..."]`. A name is defined in the file of the envoy that serves it, with a comment above it, and exported from `nnterp.components`. A family that redefines a value writes the base's name, so `layout` and `dims` cannot drift. Axis names are the constants in `nnterp/components/standard.py`, and an axis that is a fact about the model is named as the size that holds it, which lets `axis_size` resolve it.
- Every value located inside a forward carries an `unavailable=` predicate (`needs_eager`, `interface_reason`, `needs_torch_kernels`, or the family's own), so `support()` answers before a trace runs and a read raises `Unavailable` rather than `SourceNotAvailable` where the reason is known in advance. A value a family lacks is declared, `attention_probabilities = unavailable("...")`, never omitted.
- A root size a config spells its own way is `def <size>(model)` in the family module, and a model that does something of its own after the head is `def project_on_vocab(model, hidden)` there. Neither is a subclass of the root.
- One method per value in `FamilySuite`, one file per family. A new value gets a method that reads it, checks its layout and writes it, and runs on every family. A family that lacks it declares `unavailable(...)` and lists the key in its `EXPECTED_UNAVAILABLE`.
- `from __future__ import annotations`, `X | None`, `list[str]`. Raise by default, and warn only where the program can proceed.

## Workflow

1. Branch from `main`.
2. Make the change: a family module, a component value, or both, and the test that pins it.
3. Run the family's file, then the whole suite, offline and on the CPU ([Testing](testing.md)):

    ```bash
    CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 python -m pytest tests/families/test_gpt2.py -q -p no:cacheprovider
    # 42 passed, 12 skipped, 2 warnings in 5.72s
    ```

4. For a new family, add `tests/families/test_<model_type>.py` with a tiny checkpoint that is cached offline, preferring the `hf-internal-testing`, `trl-internal-testing`, `yujiepan` and `hf-tiny-v2` organizations. A config that does not parse is patched in a copy of the snapshot, the way `tests/families/test_olmo3.py` does.
5. For a new value, add a method to `FamilySuite` and expect it to run on every family.
6. Update the page under `site/docs/` that the change touches, and run every snippet on it before publishing. A page other than [transformers compatibility](transformers-compatibility.md) names no version.
7. A test that routes kernels restores `"default"` in a `finally`, since the routing is process-wide. A degenerate tiny checkpoint gets `LOAD_KWARGS` or a family-specific override, not a weaker assertion.
8. Commit as `area: lowercase imperative summary`, with a body that says why.

The skills in [ndif-team/skills](https://github.com/ndif-team/skills) state the same facts as these pages for an agent. A change that alters what an agent should do, a renamed value, a new condition on one, a new family's quirk, is a change to the matching skill as well.

## Design decisions and open items

Design decisions nnterp holds to, stated as what nnterp does not do:

- Values are on the envoy that serves them, `model.layers[i].layer_output`, not on the model as an indexed row, and there is no table of them to query apart from `support()` and `order()`.
- Nothing runs at construction except the default family's shape check, and the suite carries the causal checks. There is no load-time refusal of a block with a class not in `ENVOYS`: Such a block is a plain `Envoy` with no `layer_output`.
- Eager attention is not forced at load, and `support()` says `eager` is needed when the checkpoint runs `sdpa`.
- A remote run needs nnterp installed on the server. The family travels by reference, since an `EProperty` cannot be pickled by value.
- The kernels of a recurrent mixer are not routed at load: With `flash-linear-attention`, `causal-conv1d` or `mamba_ssm` installed the kernel values are unavailable until the caller routes them.
- The sequence axis is 2 on softmax `attention_queries`, `attention_keys` and `attention_values`, as transformers hands them to its interface, and 1 on every other value. Nothing normalizes it.
- `hasattr(envoy, name)` and `getattr(envoy, name, None)` raise `Unavailable`, and there is no `require` or `available` helper on the model that answers for a set of values at once.
- There is no wider tolerance for half precision: The suite loads each checkpoint in its own dtype and holds the contribution identity and the pattern's row sums to 8 ulp of it, and a bf16 checkpoint that needs float32 for the identity is loaded that way through `LOAD_KWARGS`.

Not done yet:

- Interventions and display helpers over the values: a logit lens over layers, patchscopes, prompt batches with targets, repeated-prompt builders, a top-k plot, a prompts-to-dataframe table. `nnterp/prompt_utils.py` and `nnterp/nnsight_utils.py` hold target-token probabilities and activation collection only, typed on `StandardizedTransformer`.
- Index lists of a hybrid's attention and recurrent blocks, and a block-structure constant per family. The blocks are read off `model.layers` with `getattr(layer, "self_attn", None)`.
- `device_map="auto"` as a default: nnsight's default stands.
- Mllama's cross-attention blocks, which need their own family and a cross-attention component. Gemma 4's and NemotronH Omni's audio towers, and video values, have no names or values. MoonViT has no family until a matching tiny checkpoint exists. `scatter_host` walks nnsight's private `_named_children`.
