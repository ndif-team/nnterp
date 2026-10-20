# Coming from nnsight

A `StandardizedTransformer` is a `TransformersModel`, so the trace is nnsight's. What changes is the block's body: the family's module paths become one set of names, and `.output` with its per-family tuple indexing becomes values that are tensors with one meaning. The nnsight trace, on GPT-2's own names:

```python
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2")

with model.trace("The Eiffel Tower is in"):
    attn = model.transformer.h[5].attn.output[0].save()   # (attn_output, attn_weights): index the tensor
    mlp = model.transformer.h[5].mlp.output.save()
    resid = model.transformer.h[5].output.save()          # a tensor on GPT-2, a tuple on BLOOM or GPT-J
    logits = model.output.logits.save()

print(attn.shape, mlp.shape, resid.shape, logits.shape)
# torch.Size([1, 7, 768]) torch.Size([1, 7, 768]) torch.Size([1, 7, 768]) torch.Size([1, 7, 50257])
```

The same trace with nnterp's names and values:

```python
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2")

with model.trace("The Eiffel Tower is in"):
    attn = model.layers[5].self_attn.attention_output.save()
    mlp = model.layers[5].mlp.mlp_output.save()
    resid = model.layers[5].layer_output.save()
    logits = model.logits.save()

print(attn.shape, mlp.shape, resid.shape, logits.shape)
# torch.Size([1, 7, 768]) torch.Size([1, 7, 768]) torch.Size([1, 7, 768]) torch.Size([1, 7, 50257])
```

The four tensors are the ones that the nnsight trace saved (the next block checks one), and this body runs unchanged on `EleutherAI/pythia-70m-deduped`, where the attention is `gpt_neox.layers[5].attention`, and on `bigscience/bloom-560m`, where the block returns a tuple and `layer_output` is still the tensor.

## What does not change

`trace`, `generate`, `.save()`, `.input`, `.output`, invokes, `tracer.iter`, `.source`, `model.edit` and `remote=True` work as [nnsight's documentation](https://nnsight.net/features/1_getting/) describes them. The native names keep working on the same model, and a trace written for a `TransformersModel` runs on a `StandardizedTransformer` as it is:

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2")

with model.trace("The Eiffel Tower is in"):
    native = model.transformer.h[5].attn.output[0].save()
    standard = model.layers[5].self_attn.attention_output.save()

print(torch.equal(native, standard), model.layers[5].self_attn is model.transformer.h[5].attn, model.layers[5].self_attn.path)
# True True model.transformer.h.5.attn
```

## nnterp's additions

### Names

`model.embed_tokens`, `model.layers[i]`, `model.layers[i].self_attn`, `model.layers[i].mlp`, `model.norm` and `model.lm_head` are nnsight aliases installed on the family's modules: an extra attribute on the same envoy, not a replacement, so `envoy.path` stays the native path. The [Names](../guide/names.md) page lists what each family's modules are called.

### Values

`layer_output`, `attention_output`, `mlp_output`, `attention_probabilities`, `logits`, `token_embeddings` and `next_token_probs` name tensors, never tuples, and each means one thing on every family. `attention_output` is what the attention adds to the residual stream even where the module returns something else. On Gemma-2 it is the post-norm's output, and on BLOOM the tensor before the residual add. The [Residual stream](../guide/values/residual-stream.md) and [Attention](../guide/values/attention.md) pages define them.

### `support()`

Not every checkpoint has every value, and `model.support()` says, from the config and before any trace, which values this checkpoint has and, for anything missing, why. Reading a missing value raises `nnterp.Unavailable` with the same reason:

```python
from nnterp import StandardizedTransformer, Unavailable

model = StandardizedTransformer("openai-community/gpt2")
print(model.support(layer=5)["self_attn.attention_probabilities"])
# read inside the eager attention forward, but this model runs 'sdpa'; load with attn_implementation='eager'

try:
    with model.trace("The Eiffel Tower is in"):
        pattern = model.layers[5].self_attn.attention_probabilities.save()
except Unavailable as error:
    print(error)
# model.transformer.h.5.attn.attention_probabilities is not available: read inside the eager attention forward, but this model runs 'sdpa'; load with attn_implementation='eager'
```

### Methods

`skip_layers`, `steer`, `project_on_vocab` and `get_topk_closest_tokens` are written against the values, so one call does the same thing on every family. The [Methods](../guide/methods.md) page has each.

The [Guide](../guide/index.md) has one page per name and value.
