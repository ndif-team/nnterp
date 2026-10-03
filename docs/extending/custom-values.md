---
title: Custom Values
one_liner: Add a new value to an attention, block or MLP by subclassing the family's envoy with an `EProperty` (keyed on a path) or a `DerivedEProperty` and passing it through `envoys=`.
tags: [extending, eproperty, envoys, source, support]
related: [docs/extending/overriding-values.md, docs/extending/finding-source-ops.md, docs/extending/registering.md, docs/extending/adding-a-family.md]
sources: [nnterp/components/eproperty.py, nnterp/components/layer.py, nnterp/components/standard.py, nnterp/components/attention.py, nnterp/standardized.py, nnterp/families/gpt2.py, tests/test_registry.py, tests/test_base.py]
---

# Custom Values

## What this is for

The standard values are the ones every family has. A value your experiment needs and
nnterp does not ship (the softmax before the dropout, the entropy of each attention row,
an MLP's output at the last position) is one descriptor on a subclass of the family's
`Attention`, `Layer` or `Mlp`, installed with `envoys=` at load. It then reads, writes
and saves like any standard value, appears in the repr with its description, and
answers `support()` on its envoy and in `model.support()`. Nothing in nnterp changes.

## Canonical pattern

Verified on `hf-internal-testing/tiny-random-gpt2`; the model id below is what a user
types.

```python
import torch
from jaxtyping import Float
from torch import Tensor

import nnterp
from nnterp import StandardizedTransformer, DerivedEProperty, EProperty
from nnterp.components import Pattern, interface_reason
from nnterp.families import gpt2
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention   # after nnterp


def attention_entropy(self) -> Float[Tensor, "batch heads query"]:
    probs = self.attention_probabilities
    return -(probs * torch.log(probs.clamp_min(1e-12))).sum(-1)


class Attention(gpt2.Attention):
    """GPT-2's attention plus two values of my own."""

    @EProperty(
        "source.attention_interface_1.source.nn_functional_softmax_0.output",
        description="The softmax output before the dropout",
        unavailable=interface_reason,
    )
    def attention_softmax(self, value) -> Pattern:
        return value

    attention_entropy = DerivedEProperty(
        attention_entropy,
        description="Per-row entropy of the pattern",
        unavailable=interface_reason,
    )


model = StandardizedTransformer("openai-community/gpt2", attn_implementation="eager", envoys={GPT2Attention: Attention})
attn = model.layers[0].self_attn

print(attn.support())
# {'attention_queries': None, ..., 'attention_softmax': None, 'attention_entropy': None}

with model.trace("The Eiffel Tower is in"):
    soft = attn.attention_softmax.save()      # [batch, heads, query, key]
    ent = attn.attention_entropy.save()       # [batch, heads, query]
```

The repr of `attn` lists both beside the standard ones:

```
(attention_softmax) -> Pattern [batch heads query key]: The softmax output before the dropout
(attention_entropy) -> Float[Tensor, 'batch heads query'] [batch heads query]: Per-row entropy of the pattern
(attention_queries) -> Queries [batch heads seq qk_head_dim]: The queries entering attention
...
```

On the tiny checkpoint `soft` is `[1, 4, 10, 10]` and `ent` is `[1, 4, 10]`, every entropy
in `[0, log(seq)]`; assigning `attn.attention_softmax` moves the logits, and under
`attn_implementation="sdpa"` both values report the same reason `attention_probabilities`
does, through `interface_reason`.

## The two descriptors

### `EProperty(key, description=, unavailable=, select=)`: a value at a path

`key` is a path from the host envoy, dotted segments ending in `output`, `input` or
`inputs`; the stub is the preprocess, mapping the served value to what the user reads.

- **`key="output"`** shares the module's `.output` location. GPT-2's attention returns
  `(attn_output, attn_weights)`, so the weights it returns are one line:

  ```python
  from nnterp import EProperty


  class Attention(gpt2.Attention):
      @EProperty(key="output", description="The attention weights the module returns beside its output")
      def returned_weights(self, value):
          return value[1]
  ```

  A preprocess that returns a slice or a copy is a view the model does not see; an
  in-place edit to it needs a `transform` to land, and an assignment needs a `postprocess`
  that rebuilds the served shape (`rewrap`). See [overriding-values.md](overriding-values.md).
  `key="input"` is the call's first argument, `key="inputs"` the raw `(args, kwargs)` pair.
- **A module named relative to the host.** A leading `../` steps to the parent, any other
  segment to a child. A path that goes up takes native names only after the `../`
  (`"../post_attention_layernorm.output"` fails on GPT-2 with `OutOfOrderError`; its native
  name is `ln_2`); one that stays at or below the host resolves aliases. On GPT-2's attention,
  `EProperty("../ln_2.output", description="The stream entering the MLP")` is the block's
  second norm; on the block, `EProperty("post_attention_layernorm.input", ...)` is the
  residual stream after the attention sublayer (verified: equal to
  `layer.input + attention_output`). In-place edits and assignments on either reach the
  model.
- **An operation inside the forward.** `source` drills into the current module's forward,
  and again into a call the forward makes; the segment after it names the operation.
  `source.attention_interface_1.source.nn_functional_softmax_0.output` is the softmax
  inside transformers' shared eager attention. Find names with
  `print(model.layers[0].self_attn.source)` ([finding-source-ops.md](finding-source-ops.md));
  the operation must exist on the path this checkpoint's forward takes, and a value inside
  the interface needs `attn_implementation="eager"`, which `unavailable=interface_reason`
  states for you. `select` picks one element: with `inputs` an int is a positional
  argument and a str a keyword, with `output` an int indexes the returned tuple.

### `DerivedEProperty(compute, description=)`: computed from other values

`compute(envoy)` runs at read time inside the trace and may read any values; its return
annotation gives the value its `layout`. The result is read-only; assigning raises
`AttributeError("attention_entropy is derived and read-only")`. Reading through other
values means forward order applies: `attention_entropy` reads the pattern, so a trace
that reads it and then `attention_queries` (bound earlier) is out of order.

## Any host: `Layer`, `Mlp`, `Standard`

A value on the block or the MLP is the same pattern on the family's other classes:

```python
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP


class Mlp(gpt2.Mlp):
    @EProperty(key="output", description="The MLP output at the last position, [batch, hidden]")
    def last_position(self, value) -> Float[Tensor, "batch hidden"]:
        return value[:, -1]


model = StandardizedTransformer("openai-community/gpt2", envoys={GPT2Attention: Attention, GPT2MLP: Mlp})
```

Subclass the family's class (`gpt2.Mlp`) rather than `nnterp.Mlp` so the family's own
overrides stay; subclass `nnterp.components.Standard` for a module that has no standard
values at all (a norm, an embedding). `Standard.values()` lists the descriptors by name,
base classes first, and `Standard.support()` their reasons. `Standard.sourced` (`False`
by default) set to `True` on a subclass instruments that envoy's forward at build, for a
value inside it that is read after the call has started (Llama 4's `Layer`, whose
`Mlp.mlp_output` is read after `attention_output`); a path alone declares where a value
is, not when its forward is instrumented.

## Layout: the return annotation

A `jaxtyping` return annotation is the value's declared shape, and the standard shapes have
names exported by `nnterp.components`: `Residual`, `Pattern`, `Keys`, ... (each defined
beside the envoy that serves it; [../usage/layouts.md](../usage/layouts.md) lists all
thirty-one: twenty-eight from `nnterp.components`, and the root's `Logits`, `NextTokenProbs`,
`Tokens` from `nnterp.standardized`).
Annotate with the name where one fits: `attention_softmax` above is `-> Pattern`, so
`Attention.attention_softmax.layout is Pattern`, the same object the base's
`attention_probabilities` carries, and `.dims` is `("batch", "heads", "query", "key")`.
A shape none of the names cover takes an inline `Float[Tensor, "..."]` with the same axis
names: `attention_entropy` is `Float[Tensor, "batch heads query"]`, so its `.dims` is
`("batch", "heads", "query")`. `isinstance(tensor, value.layout)` checks rank and dtype
either way. The family suite checks every value's tensor against its annotation and axis
sizes, so a value that will ship carries one; a value without an annotation has
`layout None` and is skipped there.

## The type key displaces the family's envoy

`envoys=` entries merge over the family's `ENVOYS`; a key given at load wins for the same
key. nnsight tries type keys before path keys, so a path key does not beat a family's type
key: `envoys={"attn": Attention}` on GPT-2 leaves the family's `gpt2.Attention` in place,
because the family already keyed `GPT2Attention`. Key yours on the type. The module
classes are in the family module's namespace (`gpt2.GPT2Attention`) or the transformers
modeling module; import them after `import nnterp`.

## Where it shows

- **The repr** of the envoy: every descriptor with a `description`, as `(name): description`.
- **`envoy.support()`**: every descriptor on the envoy's class, `None` or the reason.
- **`model.support()` and `model.support(layer=i)`**: the tree decides. The block's own
  values come from the block instance, and every child of a block that is a `Standard`
  envoy is walked under its standard name, so a custom `Attention` or `Mlp` value passed
  through `envoys=` appears as `self_attn.<name>` or `mlp.<name>`, exactly as in the
  envoy's own `support()`. With the classes above, `model.support()` gains
  `self_attn.attention_softmax`, `self_attn.attention_entropy` and `mlp.last_position`,
  and `model.support(layer=i)` the same three; a load without your `envoys=` lists none of
  them (verified on the tiny GPT-2;
  `tests/test_registry.py::test_custom_value_through_envoys_is_in_support` pins it).

## Gotchas

- **`import nnterp` before `from transformers.models... import`**; the reverse order
  segfaults at import on this stack.
- **Key on the module type, not the alias.** `envoys=` matches type, then path (native,
  or an alias where its `rename` key ends the path); a path key loses to the family's type key.
- **`unavailable=` is yours to state.** A value on an interface op without
  `unavailable=interface_reason` raises `SourceNotAvailable` at read time under `sdpa`
  instead of reporting in `support()` and raising `Unavailable` before the model runs.
- **`input` is the first argument.** A stub on an `input` path receives the tensor, not
  the pair; take an `inputs` path for a keyword argument.
- **A `DerivedEProperty`'s function takes the envoy only** (`compute(self)`), not a served
  value; it is stored as `_preprocess` so `layout` can read its annotation, but it is
  never called as a preprocess.
- **A preprocess that raises `AttributeError` is swallowed** by `Envoy.__getattr__` and
  resurfaces as "no attribute `<name>`"; raise anything else from a preprocess that can
  fail.
- **Tensors you build in the block must be on the model's device**; `torch.arange(...)` is
  on the CPU while a dispatched model is on `cuda:0`. `.to(value)` fixes it.
- **Remote traces carry the envoy classes by reference.** A class defined in a script's
  `__main__` is not importable on the server; a value that will run on NDIF lives in an
  installed module.

## Related

- [overriding-values.md](overriding-values.md): the same descriptor, relocating a standard value.
- [finding-source-ops.md](finding-source-ops.md): finding the op name for a `source.` path.
- [registering.md](registering.md): making the custom class the family's for every load of that model type in the process.
- nnsight docs/usage/extending.md and docs/developing/extending-envoy.md: the `eproperty` descriptor nnterp's are built on.
