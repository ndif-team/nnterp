# Overriding a value

When a family's forward puts a standard value somewhere other than where the base `Layer`, `Attention` or `Mlp` reads it, the family's subclass redefines the value under the same name with an `EProperty` keyed on a path from the host envoy, and the name keeps its meaning on that family too. The name, the layout annotation and the description stay; only the path changes. [EProperty's signature](../documentation/components/eproperty.md) is in the Documentation tab.

## A sibling module's output

Gemma-2's block is a sandwich, `x + post_attention_layernorm(attn(input_layernorm(x)))` and then the same around the MLP. Each sublayer adds its post-norm's output, and the family points `attention_output` and `mlp_output` at the sibling norms, and `layer_mid` and `mlp_norm_output` at the MLP's pre-norm, which is not `post_attention_layernorm` here:

```python
"""Gemma 2 (``Gemma2ForCausalLM``): Llama's names, a sandwich block."""

from transformers.models.gemma2.modeling_gemma2 import Gemma2Attention, Gemma2DecoderLayer, Gemma2MLP

from nnterp.components import Attention, EProperty, Layer, Mlp, Residual

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    @EProperty("pre_feedforward_layernorm.input", description="The residual stream after the attention add")
    def layer_mid(self, value) -> Residual:
        return value

    @EProperty("pre_feedforward_layernorm.output", description="The normed stream entering the MLP")
    def mlp_norm_output(self, value) -> Residual:
        return value


class Attention(Attention):
    @EProperty("../post_attention_layernorm.output", description="What the attention adds to the residual stream: the post-attention norm's output")
    def attention_output(self, value) -> Residual:
        return value


class Mlp(Mlp):
    @EProperty("../post_feedforward_layernorm.output", description="What the MLP adds to the residual stream: the post-feedforward norm's output")
    def mlp_output(self, value) -> Residual:
        return value


ENVOYS = {Gemma2DecoderLayer: Layer, Gemma2Attention: Attention, Gemma2MLP: Mlp}
```

With the module passed as `family=` for one load ([Registering](registering.md)), the redefined value is the norm's output and the contribution identity holds, where the module's own return would not. The smallest public Gemma-2 checkpoint is a tiny random one:

```python
import torch
from nnterp import StandardizedTransformer

import gemma2  # the module above, saved as gemma2.py beside this script

model = StandardizedTransformer("trl-internal-testing/tiny-Gemma2ForCausalLM", family=gemma2, attn_implementation="eager")
block = model.layers[0]
with model.trace("The Eiffel Tower is in"):
    x = block.layer_input.save()
    returned = block.self_attn.output.save()
    attn = block.self_attn.attention_output.save()
    normed = block.post_attention_layernorm.output.save()
    mlp = block.mlp.mlp_output.save()
    out = block.layer_output.save()
print(torch.equal(attn, normed), torch.equal(attn, returned[0]))
# True False
print(torch.equal(x + attn + mlp, out), torch.allclose(x + returned[0] + mlp, out, atol=1e-5))
# True False
```

A `../` segment steps to the parent module, and what follows it is a native name, because above the host the path is joined onto the host's own path as a string: On GPT-2's attention, `"../post_attention_layernorm.output"` names a module that does not exist and the read fails with `OutOfOrderError`, while `"../ln_2.output"` works. Below the host, and from the root after a leading `/`, aliases resolve.

| key | what it serves | a shipped use |
| --- | --- | --- |
| `"output"` | the host's own output, the same location as `.output` | the base `layer_output`, `attention_output`, `mlp_output` |
| `"input"`, `"inputs"` | the host's first argument; the whole `(args, kwargs)` pair | the base `layer_input`; the root's `input_ids` |
| `"<child>.output"`, `"<child>.input"` | a child module, aliases resolved | the base `layer_mid`, `"post_attention_layernorm.input"` |
| `"../<native>.output"` | a module named from the parent, native names only | Gemma-2's `attention_output` |
| `"/<alias>.output"`, `"/inputs"` | a path from the model's root, aliases resolved | the vision tower's `image_token_mask`, `"/inputs"` |
| `"source.<op>.input"`, `"source.<op>.output"` | an operation of the host's forward | BLOOM's `attention_output`, `"source.dropout_add_0.input"` |
| `"source.<call>.inputs"` with `select=` | one argument of a call the forward makes | the base `attention_queries`, `select=1` |
| `"source.<call>.source.<op>.output"` | an operation inside a function the forward calls | the base `attention_probabilities` |
| a function of the host | a path decided at read time, inside the trace | Falcon's `by_alibi(...)`, one operation per `alibi` setting |

## An operation inside the forward

BLOOM's sublayers take the residual as an argument and add it inside the module (`dropout_add(x, residual, ...)`). The module's output is then a state of the residual stream, and the contribution is the first argument of that call:

```python
# nnterp/families/bloom.py
class Attention(Attention):
    @EProperty(
        "source.dropout_add_0.input",
        description="What the attention adds to the residual stream: the tensor entering dropout_add",
    )
    def attention_output(self, value) -> Residual:
        return value
```

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("hf-internal-testing/tiny-random-BloomForCausalLM")
block = model.layers[0]
with model.trace("The Eiffel Tower is in"):
    x = block.layer_input.save()
    attn = block.self_attn.attention_output.save()   # inside the attention's forward: before the module's own output
    returned = block.self_attn.output.save()
    mlp = block.mlp.mlp_output.save()
    out = block.layer_output.save()
print(torch.equal(returned[0], x + attn), torch.equal(x + attn + mlp, out))
# True True
```

The segment after `source` is the operation's label in `print(envoy.source)` ([Finding source ops](finding-source-ops.md)). The descriptor walks the path before every read or write, which instruments the operation for the current run. A value inside a forward is read before the module's own `.output`, as the block above does. In the other order the read fails with `OutOfOrderError`. With `inputs`, `select=n` picks a positional argument and `select="name"` a keyword one, and with `output`, `select=i` indexes a returned tuple. A write repacks the element, and assigning one argument replaces that argument alone. The base's `attention_queries` is `EProperty(f"source.{INTERFACE}.inputs", select=1, ...)`, the second argument of the shared attention call, and BLOOM's is `EProperty("source.self__reshape_0.output", select=0, description=Attention.attention_queries.description)`, the first element of the tuple its `_reshape` returns. `Attention.attention_queries` on the class is the descriptor itself, which lets the redefinition reuse the base's description, and `-> Queries` keeps the base's layout.

## A value the family does not have

A marker in the class body takes the place of the inherited descriptor. It keeps the name in the tree and in the repr, as `(layer_mid): Unavailable: <reason>`, makes `support()` report the reason, and makes any read or write raise `nnterp.Unavailable` with it before the model runs. Cohere's block is parallel and has no stream between its sublayers:

```python
# nnterp/families/cohere.py
class Layer(Layer):
    layer_mid = unavailable(PARALLEL)
```

When the checkpoint decides, `unavailable=` takes a function of the envoy returning a reason or `None`. Every value the base `Attention` reads inside the shared attention forward takes `unavailable=interface_reason`, which calls the envoy's `off_interface()`. The default answer is `needs_eager`, the reason given when the model runs `sdpa`. A family with another reason overrides the method once rather than each value. GPT-2's `reorder_and_upcast_attn` routes around the interface:

```python
# nnterp/families/gpt2.py
class Attention(Attention):
    def off_interface(self):
        if self._module.config.reorder_and_upcast_attn:
            return "this checkpoint sets reorder_and_upcast_attn, which takes GPT-2's own upcast attention path"
        return super().off_interface()
```

```python
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", attn_implementation="eager")
print(model.support()["self_attn.attention_scores"])
# None
model.config.reorder_and_upcast_attn = True
print(model.support()["self_attn.attention_scores"][0])
# this checkpoint sets reorder_and_upcast_attn, which takes GPT-2's own upcast attention path
```

A predicate must not raise `AttributeError`. The descriptor re-raises one as a `RuntimeError` naming the check, because a descriptor's `AttributeError` would otherwise come back as "no attribute" and hide the predicate's own bug.

## A value that has to be a copy

Falcon's parallel block adds the attention output into the MLP's output tensor in place, and a plain `mlp_output` would change under a saved read. The value reads a clone, and because a clone is invisible to the model, a `transform` hands the edited copy back to be swapped in once the block is done with the read:

```python
# nnterp/families/falcon.py
from nnterp.components import EProperty, Mlp, Residual, first_tensor, rewrap


class Mlp(Mlp):
    @EProperty(key="output", description="What the MLP adds to the residual stream (a copy, since the block adds the attention into the live tensor in place)")
    def mlp_output(self, value) -> Residual:
        return first_tensor(value).clone()

    @mlp_output.postprocess
    def mlp_output(self, value):
        return rewrap(self, value)

    @mlp_output.transform
    def mlp_output(self, value, raw):
        return value.clone()
```

The three hooks answer three different events. The decorated function is the getter, and what it returns is what a read serves. `postprocess` runs on an assigned value, and what it returns is what is written to the location. `transform` runs on the served view after an in-place edit, and what it returns is what the model receives when the block moves on.

`postprocess` is the write side of a value keyed on `"output"`: The base's `layer_output` and `mlp_output` read `first_tensor(value)` and write `rewrap(self, value)`, both from `nnterp.components`, where `first_tensor` is the tensor of a module output, its first element when the output is a tuple, and `rewrap` puts a tensor back into that tuple, so a module that returns a tuple reads as a tensor and an assignment puts the tensor back into its tuple. A value redefined on a path that serves a bare tensor, such as a sibling norm's output or an operation's, needs neither. nnsight's [eproperty](https://nnsight.net/documentation/intervention/eproperty/) reference has `postprocess` and `transform` in full. A root size the config spells its own way is not a descriptor override but a module-level function named after the size ([Adding a family](adding-a-family.md)), and a value that is new rather than relocated is [Custom values](custom-values.md).
