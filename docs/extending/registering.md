---
title: Registering a Family
one_liner: `nnterp.families.register(module)` adds a family from outside the package or overrides a shipped one, process-wide, names, envoy classes and size functions included; `StandardizedTransformer(..., family=module)` uses one for a single load, and `rename=`/`envoys=` layer extra names and envoys on top.
tags: [extending, families, registry, lookup, family]
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
The per-model alternatives are `family=`, which skips the lookup for one load
([below](#passing-a-family-at-load)), and `rename=` and `envoys=`, which layer on top of
whichever family the load uses.

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

## Passing a family at load

`family=` hands one load its family directly. The lookup is skipped (the config is not
read for it) and `REGISTRY` is not touched, so other loads of the same model type keep
theirs. The usual form is a module of your own, written like a shipped one
([adding-a-family.md](adding-a-family.md)):

```python
from nnterp import StandardizedTransformer, families
from nnterp.families import gpt2

import my_family                    # RENAME, ENVOYS, the classes, any size functions

model = StandardizedTransformer("openai-community/gpt2", family=my_family)
assert model.family is my_family
assert families.lookup("gpt2") is gpt2     # every other load is unchanged
```

A family built in code is a `types.SimpleNamespace`; nnterp reads it with attribute access
only, the same as a module. It needs `RENAME` and `ENVOYS`; `MODEL_TYPES` is only read by
`register` and is not checked against the checkpoint, so a family can be applied to a
checkpoint whose `model_type` it does not name (a fork with the same module classes under
a new type, say). Functions named after a root size or `project_on_vocab` win over the
root's rule exactly as on a shipped family.

Extending a shipped family is the common case. `vars(module)` copies everything the
module defines, its size functions included, so only what changes is written:

```python
import types

from nnterp import StandardizedTransformer
from nnterp.components import DerivedEProperty
from nnterp.families import llama
from transformers.models.llama.modeling_llama import LlamaMLP   # after nnterp


class Mlp(llama.Mlp):
    width = DerivedEProperty(lambda self: self._module.intermediate_size, description="The hidden width")


family = types.SimpleNamespace(**vars(llama))
family.RENAME = {**llama.RENAME, "post_attention_layernorm": ["post_attention_layernorm", "ln2"]}
family.ENVOYS = {**llama.ENVOYS, LlamaMLP: Mlp}

model = StandardizedTransformer("HuggingFaceTB/SmolLM2-135M", family=family)
assert type(model.layers[0].mlp) is Mlp
assert model.layers[0].ln2 is model.layers[0].post_attention_layernorm
model.support()["mlp.width"]       # None: available on every block
```

A function the shipped family does not define goes straight into the constructor,
`types.SimpleNamespace(**vars(gpt2), num_kv_heads=lambda model: 1)`; one it does define
(`RENAME`, `ENVOYS`, GPT-2's `intermediate_size`) is set afterwards as above, since the
constructor refuses a keyword given twice.

`rename=` and `envoys=` still apply on top: the envoys are nnsight's tensor-parallel
defaults, then the family's `ENVOYS`, then `envoys=`; the aliases are the family's
`RENAME`, then `rename=`.

## The lookup order

1. `REGISTRY[model_type]`, if `register` put one there.
2. `nnterp.families.<model_type>`, imported on first use.
3. `UnsupportedFamily`, listing what exists:

```
UnsupportedFamily: no standardization for model_type 'zamba'; known: ['afmoe', 'apertus', ..., 'zaya'].
Add nnterp/families/zamba.py with MODEL_TYPES, RENAME and ENVOYS, or pass a module to nnterp.families.register().
```

The list is `sorted(set(known()) | set(REGISTRY))`, so a registered type appears there.

- `families.known()` is the shipped modules' names, from the package directory, without
  importing them. It does not list registered families.
- `families.all_families()` imports and returns every shipped module; for tooling and
  tests, not for a load. It does not include registered families.
- `families.<model_type>` (`families.gpt2`, `families.qwen3_5_text`) is the shipped module,
  imported on first attribute access; `dir(families)` lists them. A name that is not a
  shipped module raises `AttributeError`.
- `model.family` is the object the load resolved to, registered or shipped.

## `register` versus `family=` versus `rename=` / `envoys=`

| | `register(family)` | `family=` on a load | `rename=` / `envoys=` on a load |
| --- | --- | --- | --- |
| scope | every load of those model types in this process | that one model | that one model |
| what changes | the whole family: names, envoy classes and size functions, for every load | the whole family, for this load | extra aliases merged over the family's `RENAME`; extra envoy classes merged over its `ENVOYS`; a key given wins |
| needs `MODEL_TYPES` | yes | no | no |
| `model.family` | the registered object | the object passed | the family the load resolved to |
| `model.support()` | the values on the tree the registered `ENVOYS` build | the values on the tree the passed `ENVOYS` build | the values on the tree, including any a class passed through `envoys=` adds |
| undo | `del families.REGISTRY[model_type]` | load again without it | load again without it |

```python
model = StandardizedTransformer("openai-community/gpt2", rename={"mlp": "ffn"})
assert model.layers[0].ffn is model.layers[0].mlp
assert model.family is gpt2
```

`envoys=` matches by module type or native path, never by alias, and type keys are tried
before path keys, so displacing a family's type-keyed envoy takes a type key of your own
([custom-values.md](custom-values.md)).

## Gotchas

- **Register before loading.** `lookup` runs in `StandardizedTransformer.__init__`; a
  model built earlier keeps the family it resolved to.
- **`register` overrides silently.** Registering `("gpt2",)` makes every later GPT-2 load
  in the process yours; a test that registers cleans up with `del families.REGISTRY[...]`
  in a `finally`.
- **`Layer`, `Attention`, `Mlp` on the namespace are optional.** `support()` walks the
  tree and reads none of them; the `UnsupportedFamily` message names the three attributes
  the load path reads, and they are enough. The suite (`FamilySuite`) does read the classes.
- **`known()` and `all_families()` are the shipped modules only.**
- **Carry a shipped family's size functions into a variant.** `RENAME` and `ENVOYS` are
  dicts to spread; `num_kv_heads`, `head_dim`, `qk_head_dim` and `intermediate_size` are
  module functions the root looks up on `model.family` by name, so a variant of Falcon,
  DeepSeek, GPT-2, GPT-J, OPT, MPT or BLOOM passes them on
  (`intermediate_size=gpt2.intermediate_size`, or start from `vars(gpt2)`) or the root's
  plain rule answers.
- **A registered or passed family's classes must be importable by name where the trace runs.** A
  remote trace carries the envoy tree's classes by reference; a class defined in a
  script's `__main__` or a `SimpleNamespace` built inline is not importable on a server.
  For local use it is fine.
- **`route_delta_rule` on a registered hybrid** reads the mixer class off `family.ENVOYS`;
  keep the `LinearAttention` subclass keyed on the transformers mixer type.

## Related

- [adding-a-family.md](adding-a-family.md): the module `register` and `family=` take, and its test file.
- [custom-values.md](custom-values.md): a value on one load through `envoys=`; `model.support()` lists it either way.
- [overriding-values.md](overriding-values.md): what to change in a variant's classes.
