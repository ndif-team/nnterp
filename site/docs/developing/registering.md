# Registering

`nnterp.families.register(family, *model_types)` makes a family the one every later load of those model types resolves to, for the rest of the process, whether the family is kept outside the package or replaces a shipped one. `StandardizedTransformer(repo_id, family=module)` uses a family for one load and leaves the registry alone. Either takes any module or object with `RENAME` and `ENVOYS`. The signatures of `register` and `lookup` are under [families](../documentation/families/index.md) in the Documentation tab.

```python
import types

from nnterp import StandardizedTransformer, families
from nnterp.families import gpt2

variant = types.SimpleNamespace(**vars(gpt2))                       # everything the shipped module defines
variant.RENAME = {**gpt2.RENAME, "mlp": ["mlp", "ffn"]}            # one more alias for the feed-forward
families.register(variant, "gpt2")

model = StandardizedTransformer("openai-community/gpt2")
print(model.family is variant, model.layers[0].ffn is model.layers[0].mlp, model.intermediate_size)
# True True 3072

del families.REGISTRY["gpt2"]                                      # back to the shipped module
print(families.lookup("gpt2") is gpt2)
# True
```

`vars(gpt2)` copies everything the shipped module defines, and the variant carries GPT-2's `intermediate_size` function along with its classes. A namespace built from `RENAME` and `ENVOYS` alone also loads and answers `support()` with every block value, because `support()` walks the envoy tree the `ENVOYS` build and reads no class off the family. The root's sizes and `project_on_vocab`, though, look for a function of their name on `model.family` before falling back to their own rule, and with that namespace `model.intermediate_size` falls back to `config.intermediate_size`, which GPT-2's config does not have, and raises `AttributeError`. A variant of a family that defines a size function, such as GPT-2, Falcon, BLOOM or DeepSeek-V2, passes it on, which the spread above does.

## One load

`family=` skips the lookup, leaving the config unread for it and `REGISTRY` untouched:

```python
import types

from nnterp import StandardizedTransformer, families
from nnterp.families import gpt2

variant = types.SimpleNamespace(**vars(gpt2))
variant.RENAME = {**gpt2.RENAME, "mlp": ["mlp", "ffn"]}

model = StandardizedTransformer("openai-community/gpt2", family=variant)
print(model.family is variant, model.layers[0].ffn is model.layers[0].mlp)
# True True
print(families.lookup("gpt2") is gpt2, families.REGISTRY)
# True {}
```

No model type is named or checked against the checkpoint, and a family can be applied to a checkpoint with a `model_type` it was not written for. `rename=` and `envoys=` still apply on top of either route: The aliases are the family's `RENAME` and then `rename=`, the envoys nnsight's tensor-parallel defaults, then the family's `ENVOYS`, then `envoys=`, and a key given later wins.

## A module of your own

A family written as a file ([Adding a family](adding-a-family.md)) and named after its model type registers with no types given: `register` takes the type from the last component of the module's `__name__`, as a shipped module's file name gives it.

```python
from nnterp import StandardizedTransformer, families

import llama  # a family module of my own, named after the model_type it covers

families.register(llama)                        # no types given: covers "llama", from the module's name
print(families.lookup("llama") is llama)
# True
model = StandardizedTransformer("HuggingFaceTB/SmolLM2-135M")
print(model.family is llama)
# True
del families.REGISTRY["llama"]
```

A module named otherwise, or one covering several types, names them, `families.register(my_family, "zamba", "zamba2")`. A `SimpleNamespace` built from scratch has no `__name__`, and `register(ns)` raises `TypeError: register() needs the model types the family covers: a SimpleNamespace has no __name__ to take one from`. One spread from `vars(gpt2)` inherits `nnterp.families.gpt2` as its `__name__` and registers under `gpt2`.

## The lookup order

`lookup(model_type)` returns, in order, `REGISTRY[model_type]` if `register` put one there, else `nnterp.families.<model_type>` imported on first use, else `nnterp.families.default`, the best-effort family, with this warning:

```
UserWarning: nnterp has no family for model_type 'nanochat'; the default family standardizes it as a best-effort guess.
Check model.support() for what it found, and add nnterp/families/nanochat.py (or nnterp.families.register(family, 'nanochat'))
for a standardization you can rely on.
```

Registering a family for the type, or shipping its module, takes it off the default. Register before loading: The lookup runs in `StandardizedTransformer.__init__`, and a model built earlier keeps the family it resolved to. `register` overrides silently, and a test that registers cleans up with `del families.REGISTRY[...]` in a `finally`. `families.known()` lists the shipped modules' names only, not registered families and not `default`. `model.family` is whatever the load resolved to. Another engine has its own registry: `register(family, *model_types, engine="vllm")` registers a family for `StandardizedVLLM`, which `lookup(model_type, engine="vllm")` finds, and a family registered without `engine` is a transformers family only.
