# Quick Start

Everything here runs on CPU with GPT-2.

## Load a model

```python
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2")
```

`StandardizedTransformer` is an nnsight `TransformersModel`. It reads the checkpoint's
`model_type`, picks the matching family, and gives the envoy tree the standard names. Any keyword
`TransformersModel` accepts is accepted here.

## The standard names

Every family answers to the same names:

```python
model.embed_tokens
model.layers[i]
model.layers[i].self_attn
model.layers[i].mlp
model.norm
model.lm_head
```

They are aliases. On GPT-2, `model.layers[5]` and `model.transformer.h[5]` are the same envoy,
and the native names keep working.

## The standard values

A name says which module. A value says which tensor, and it means the same thing on every
family:

```python
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2")

with model.trace("The Eiffel Tower is in"):
    attn = model.layers[5].self_attn.attention_output.save()   # what attention adds to the residual stream
    mlp = model.layers[5].mlp.mlp_output.save()                # what the MLP adds
    resid = model.layers[5].layer_output.save()                # the residual stream leaving block 5
    logits = model.logits.save()

print(attn.shape, mlp.shape, resid.shape, logits.shape)
# torch.Size([1, 7, 768]) torch.Size([1, 7, 768]) torch.Size([1, 7, 768]) torch.Size([1, 7, 50257])
```

`layer_output` is a tensor whether the family's block returns a tensor or a tuple.
`attention_output` and `mlp_output` are contributions, defined by one identity that holds on
every family:

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2")

with model.trace("The Eiffel Tower is in"):
    block = model.layers[5]
    x = block.input.save()
    a = block.self_attn.attention_output.save()
    m = block.mlp.mlp_output.save()
    out = block.layer_output.save()

print(torch.allclose(x + a + m, out, atol=1e-4))   # True
```

Values can be read, edited in place, or assigned, like any nnsight `.output`.

## Ask before you trace

Not every checkpoint has every value. `model.support()` says what this one has and, for anything
missing, why. Reading a missing value inside a trace raises `nnterp.Unavailable` with the same
reason.

```python
print(model.support(layer=5)["self_attn.attention_probabilities"])
# read inside the eager attention forward, but this model runs 'sdpa'; load with attn_implementation='eager'
```

Anything inside attention needs the eager implementation:

```python
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", attn_implementation="eager")

with model.trace("The Eiffel Tower is in"):
    pattern = model.layers[3].self_attn.attention_probabilities.save()   # [batch, num_heads, query, key]

print(pattern.shape)
# torch.Size([1, 12, 7, 7])
```

## Methods

A few operations are common enough to be methods on the model:

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2")
vector = torch.randn(model.hidden_size)

with model.trace("The Eiffel Tower is in"):
    lens = model.project_on_vocab(model.layers[6].layer_output).save()  # logit lens at block 6
    model.steer(8, vector, factor=3, token_positions=-1)                # add to block 8's output
    model.skip_layers(10, 11)                                           # blocks 10..11 do not run
    logits = model.logits.save()
```

## Next

The [Documentation](../documentation/index.md) tab is the API reference, one page per module.
