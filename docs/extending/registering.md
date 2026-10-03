---
title: Registering a Family
one_liner: `nnterp.families.register(module)` adds a family from outside the package or overrides a shipped one, process-wide, names, envoy classes and size functions included; `rename=`/`envoys=` at load are the per-model alternative.
tags: [extending, families, registry, lookup]
related: [docs/extending/adding-a-family.md, docs/extending/custom-values.md, docs/extending/overriding-values.md]
sources: [nnterp/families/__init__.py, nnterp/standardized.py, tests/test_registry.py]
---

# Registering a Family

## What this is for

`StandardizedTransformer` resolves a checkpoint's `config.model_type` through
`nnterp.families.lookup`, which consults `REGISTRY` first and the module named after the
type second. `register(family)` puts a family into `REGISTRY`, so a family kept in your
own package, or a variant of a shipped one, is what every load of that type resolves to,
without a file under `nnterp/families/`. It is process-wide, like installing a kernel.
The per-model alternatives are `rename=` and `envoys=` on one load.

## Canonical pattern

Verified on `hf-internal-testing/tiny-random-gpt2`: a variant of the shipped GPT-2 family
that marks the pattern unavailable and adds an alias.

```python
import types

import nnterp
from nnterp import StandardizedTransformer, families
from nnterp.families import gpt2
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention   # after nnterp


class Attention(gpt2.Attention):
    attention_probabilities = nnterp.unavailable("disabled by the registered variant")


variant = types.SimpleNamespace(
    MODEL_TYPES=("gpt2",),
    RENAME={**gpt2.RENAME, "mlp": ["mlp", "ffn"]},
    ENVOYS={**gpt2.ENVOYS, GPT2Attention: Attention},
    Layer=gpt2.Layer, Attention=Attention, Mlp=gpt2.Mlp,   # optional: support() walks the tree
    intermediate_size=gpt2.intermediate_size,               # GPT-2's size spelling (n_inner), or the root's plain rule answers
)
families.register(variant)

model = StandardizedTransformer("openai-community/gpt2", attn_implementation="eager")
assert model.family is variant
assert type(model.layers[0].self_attn) is Attention
assert model.layers[0].ffn is model.layers[0].mlp is model.transformer.h[0].mlp
model.support()["self_attn.attention_probabilities"]
# {0: 'disabled by the registered variant', 1: ..., ...}

del families.REGISTRY["gpt2"]          # back to the shipped module
assert families.lookup("gpt2") is gpt2
```

A family module written as a file ([adding-a-family.md](adding-a-family.md)) registers the
same way: `families.register(my_package.my_family)`. `register` returns the family, so it
works as a decorator-style one-liner at the module's end or at import of your package.

## What a registered family needs

`register` takes any module or object with `MODEL_TYPES`, `RENAME` and `ENVOYS`, and
those three are all a load needs: `types.SimpleNamespace(MODEL_TYPES=("gpt2",),
RENAME=gpt2.RENAME, ENVOYS=gpt2.ENVOYS)` loads, traces, and answers `support()` with every
block value (`tests/test_registry.py::test_register_needs_only_names_and_envoys_for_support`).
`StandardizedTransformer.support()` walks the envoy tree the `ENVOYS` build, so the
`self_attn.*` / `mlp.*` / `linear_attn.*` names are whatever the blocks' `Standard`
children carry; the family's `Attention`, `Mlp` and `LinearAttention` attributes are not
read.

The root's sizes read the family too: each is a `StandardizedProperty` that calls
`getattr(model.family, <name>)` with the model when it exists, else its plain rule over the
config. So a registered family may carry a function named after any root size, and the
same `SimpleNamespace` with `hidden_size=lambda model: 999` makes `model.hidden_size` 999
while `model.num_heads` keeps the root's `config.num_attention_heads`
(`tests/test_registry.py::test_family_defines_a_size_instead_of_the_root`). What a
shipped family spells its own way is listed in
[../usage/root-values.md](../usage/root-values.md#sizes).

Two things do read more of the family later:

- `nnterp.route_delta_rule(model.family, ...)` finds a hybrid's mixer module through
  `family.ENVOYS`.
- The suite's `test_envoy_classes` and `expected_values` read `FAMILY.Layer`,
  `FAMILY.Attention`, `FAMILY.Mlp` and `FAMILY.LinearAttention`, so a variant run through
  `FamilySuite` carries the classes.

A shipped family's module has all of these, so a variant built by spreading a shipped
family's dicts, reusing its classes and carrying its size functions, as above, is
complete. The size functions are the part a spread of `RENAME` and `ENVOYS` leaves
behind: on `hf-internal-testing/tiny-random-gpt2` a variant without
`intermediate_size=gpt2.intermediate_size` answers `model.intermediate_size` with the
config's unused `intermediate_size` key (37) instead of `n_inner`'s `4 * hidden_size`
(128).

## The lookup order

1. `REGISTRY[model_type]`, if `register` put one there.
2. `nnterp.families.<model_type>`, imported on first use.
3. `nnterp.families.default`, the best-effort family, with a warning; it checks its guess at
   load and raises `UnsupportedFamily` when it cannot standardize the checkpoint
   ([loading.md](../usage/loading.md#an-architecture-with-no-family)).

```
UserWarning: nnterp has no family for model_type 'zamba'; the default family standardizes it as a
best-effort guess. Check model.support() for what it found, and add nnterp/families/zamba.py (or
nnterp.families.register()) for a standardization you can rely on.
```

Registering a family for the type, or shipping its module, takes it off the default.

- `families.known()` is the shipped modules' names, from the package directory, without
  importing them. It does not list registered families, nor `default`.
- `families.all_families()` imports and returns every shipped module; for tooling and
  tests, not for a load. It does not include registered families.
- `families.<model_type>` (`families.gpt2`, `families.qwen3_5_text`) is the shipped module,
  imported on first attribute access; `dir(families)` lists them. A name that is not a
  shipped module raises `AttributeError`.
- `model.family` is the object the load resolved to, registered or shipped.

## `register` versus `rename=` / `envoys=`

| | `register(family)` | `rename=` / `envoys=` on a load |
| --- | --- | --- |
| scope | every load of those model types in this process | that one model |
| what changes | the whole family: names, envoy classes and size functions, for every load | extra aliases merged over the family's `RENAME`; extra envoy classes merged over its `ENVOYS`; a key given wins |
| `model.family` | the registered object | the shipped module |
| `model.support()` | the values on the tree the registered `ENVOYS` build | the values on the tree, including any a class passed through `envoys=` adds |
| undo | `del families.REGISTRY[model_type]` | load again without it |

```python
model = StandardizedTransformer("openai-community/gpt2", rename={"mlp": "ffn"})
assert model.layers[0].ffn is model.layers[0].mlp
assert model.family is gpt2
```

`envoys=` matches by module type or path, and type keys are tried before path keys, so
displacing a family's type-keyed envoy takes a type key of your own
([custom-values.md](custom-values.md)).

## Gotchas

- **Register before loading.** `lookup` runs in `StandardizedTransformer.__init__`; a
  model built earlier keeps the family it resolved to.
- **`register` overrides silently.** Registering `("gpt2",)` makes every later GPT-2 load
  in the process yours; a test that registers cleans up with `del families.REGISTRY[...]`
  in a `finally`.
- **`Layer`, `Attention`, `Mlp` on the namespace are optional.** `support()` walks the
  tree and reads none of them; `MODEL_TYPES`, `RENAME` and `ENVOYS` are what the load path
  reads, and they are enough. The suite (`FamilySuite`) does read the classes.
- **`known()` and `all_families()` are the shipped modules only.**
- **Carry a shipped family's size functions into a variant.** `RENAME` and `ENVOYS` are
  dicts to spread; `num_kv_heads`, `head_dim`, `qk_head_dim` and `intermediate_size` are
  module functions the root looks up on `model.family` by name, so a variant of Falcon,
  DeepSeek, GPT-2, GPT-J, OPT, MPT or BLOOM passes them on
  (`intermediate_size=gpt2.intermediate_size`) or the root's plain rule answers.
- **A registered family's classes must be importable by name where the trace runs.** A
  remote trace carries the envoy tree's classes by reference; a class defined in a
  script's `__main__` or a `SimpleNamespace` built inline is not importable on a server.
  For local use it is fine.
- **`route_delta_rule` on a registered hybrid** reads the mixer class off `family.ENVOYS`;
  keep the `LinearAttention` subclass keyed on the transformers mixer type.

## Related

- [adding-a-family.md](adding-a-family.md): the module `register` takes, and its test file.
- [custom-values.md](custom-values.md): a value on one load through `envoys=`; `model.support()` lists it either way.
- [overriding-values.md](overriding-values.md): what to change in a variant's classes.
