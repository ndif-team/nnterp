# Custom values

A value nnterp does not ship is one descriptor on a subclass of the family's `Attention`, `Layer` or `Mlp`, installed for one load with `envoys=`: an `EProperty` keyed on a path from the host, or a `DerivedEProperty` computed from other values. It then reads, writes and saves like a standard value, is listed in the repr with its description, and answers `support()`.

```python
import torch
from jaxtyping import Float
from torch import Tensor

from nnterp import StandardizedTransformer, DerivedEProperty, EProperty
from nnterp.components import BATCH, HEADS, QUERY, Pattern, interface_reason
from nnterp.families import gpt2
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention  # after nnterp


def attention_entropy(self) -> Float[Tensor, f"{BATCH} {HEADS} {QUERY}"]:
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

    attention_entropy = DerivedEProperty(attention_entropy, description="Per-row entropy of the pattern", unavailable=interface_reason)


model = StandardizedTransformer("openai-community/gpt2", attn_implementation="eager", envoys={GPT2Attention: Attention})
attn = model.layers[0].self_attn
print([line.strip() for line in repr(attn).splitlines() if "attention_softmax" in line or "attention_entropy" in line])
# ['(attention_softmax) -> Pattern [batch num_heads query key]: The softmax output before the dropout',
#  "(attention_entropy) -> Float[Tensor, 'batch num_heads query'] [batch num_heads query]: Per-row entropy of the pattern"]
print(attn.support()["attention_softmax"], model.support()["self_attn.attention_entropy"])
# None None
with model.trace("The Eiffel Tower is in"):
    soft = attn.attention_softmax.save()
    ent = attn.attention_entropy.save()
print(soft.shape, ent.shape)
# torch.Size([1, 12, 7, 7]) torch.Size([1, 12, 7])
```

The softmax is an operation inside transformers' shared eager attention, and the path has a `source` segment for the call and another for the operation inside it ([Finding source ops](finding-source-ops.md)). `unavailable=interface_reason` is the predicate the base's interior values carry: Under the checkpoint's default `sdpa`, both values then report "read inside the eager attention forward, but this model runs 'sdpa'; load with attn_implementation='eager'" in `support()` instead of failing at read time with `SourceNotAvailable`.

A `DerivedEProperty` runs its function on the envoy at read time, inside the trace, and forward order applies to what it reads: `attention_entropy` reads the pattern, and a trace that reads it and then `attention_queries` is out of order. It is read-only, and assigning raises `AttributeError: attention_entropy is derived and read-only`.

## On the block or the MLP

The same pattern on the family's other classes, keyed on the module's own output and on a child module:

```python
from jaxtyping import Float
from torch import Tensor

from nnterp import StandardizedTransformer, EProperty
from nnterp.components import BATCH, HIDDEN, Residual
from nnterp.families import gpt2
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2MLP  # after nnterp


class Mlp(gpt2.Mlp):
    @EProperty(key="output", description="The MLP output at the last position")
    def last_position(self, value) -> Float[Tensor, f"{BATCH} {HIDDEN}"]:
        return value[:, -1]


class Layer(gpt2.Layer):
    @EProperty("ln_2.input", description="The residual stream after the attention sublayer")
    def mid_stream(self, value) -> Residual:
        return value


model = StandardizedTransformer("openai-community/gpt2", envoys={GPT2MLP: Mlp, GPT2Block: Layer})
with model.trace("The Eiffel Tower is in"):
    mid = model.layers[0].mid_stream.save()
    last = model.layers[0].mlp.last_position.save()
print(mid.shape, last.shape)
# torch.Size([1, 7, 768]) torch.Size([1, 768])
print([name for name in model.support() if name in ("mid_stream", "mlp.last_position")])
# ['mid_stream', 'mlp.last_position']
```

`key="output"` shares the module's own output location, and a stub that returns a slice hands the reader a view the model does not see: An in-place edit to `last_position` needs a `transform` to land, and an assignment a `postprocess` that rebuilds the served shape ([Overriding a value](overriding-values.md)). A path below the host resolves aliases, and `"ln_2.input"` and `"post_attention_layernorm.input"` are the same location on GPT-2's block, where the base reads `layer_mid`. Above the host, after `../`, only native names resolve. `model.support()` walks each block's children that carry values under their standard names, where a value on an `Attention` or `Mlp` appears as `self_attn.<name>` or `mlp.<name>`, and a load without your `envoys=` lists none of them.

## The layout annotation

The return annotation is the value's declared layout, read by `.layout` and `.dims` and shown in the repr. The standard layouts, such as `Residual`, `Pattern` and `Keys`, are exported by `nnterp.components`, and a shape none of them has takes an inline `Float[Tensor, ...]` composed from the same axis constants, such as `BATCH`, `HEADS`, `QUERY` and `HIDDEN`. `attention_softmax` above is annotated `-> Pattern`, and `Attention.attention_softmax.layout is Pattern`, the object the base's `attention_probabilities` carries. `Attention.attention_entropy.dims` is `('batch', 'num_heads', 'query')`. The family suite checks every value's tensor against its annotation, which a value that will ship therefore carries. Without an annotation `layout` is `None`, and the repr lists `(name): description`.

## Key on the module type

`envoys=` entries merge over the family's `ENVOYS`, and nnsight tries type keys before path keys: `envoys={"attn": Attention}` on GPT-2 leaves the family's `gpt2.Attention`, keyed on `GPT2Attention`, in place. Key yours on the type, and subclass the family's class rather than `nnterp.Attention` so the family's own overrides stay. To make the class the family's for every load of the model type in the process, [Registering](registering.md) is the next page.
