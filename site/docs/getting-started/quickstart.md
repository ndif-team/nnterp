# Quick Start

A `StandardizedTransformer` is an nnsight `TransformersModel` with one set of module names on every family, and with values on its blocks that mean the same thing everywhere. Load GPT-2 and read block 5's values and the logits:

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

The constructor reads the checkpoint's `model_type`, picks the matching family, and passes every other keyword to `TransformersModel`.

## The names

`model.embed_tokens`, `model.layers[i]`, `model.layers[i].self_attn`, `model.layers[i].mlp`, `model.norm` and `model.lm_head` exist on every family. They are nnsight aliases, so the native names keep working and both reach the same envoy:

```python
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2")

print(model.layers[5] is model.transformer.h[5], model.layers[5].self_attn is model.transformer.h[5].attn)
# True True
```

## The values

A name says which module; a value says which tensor. `layer_output` is a tensor whether the family's block returns a tensor or a tuple, and `attention_output` and `mlp_output` are what each sublayer adds to the residual stream, wherever the family adds it, so one identity holds on every family:

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

print(torch.allclose(x + a + m, out, atol=1e-4))
# True
```

A value is read, edited in place or assigned the way an nnsight `.output` is.

## Ask before you trace

Not every checkpoint has every value. `model.support()` says, from the config alone, what this one has and, for anything missing, why. Reading a missing value inside a trace raises `nnterp.Unavailable` with the same reason, before the model runs:

```python
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2")

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

The operations an experiment repeats are methods on the model, written against the values, so they run on every family:

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2")
torch.manual_seed(0)
vector = torch.randn(model.hidden_size)

with model.trace("The Eiffel Tower is in"):
    lens = model.project_on_vocab(model.layers[6].layer_output).save()  # logit lens at block 6
    model.steer(8, vector, factor=3, token_positions=-1)                # add to the stream leaving block 8
    model.skip_layers(10, 11)                                           # blocks 10 and 11 do not run
    logits = model.logits.save()

print(model.tokenizer.decode(lens[0, -1].argmax()), logits.shape)
#  front torch.Size([1, 7, 50257])
```

The [Guide](../guide/index.md) has one page per name and value, and [Coming from nnsight](from-nnsight.md) sets the same trace beside its nnsight form.
