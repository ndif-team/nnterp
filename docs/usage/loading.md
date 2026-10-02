---
title: Loading a model
one_liner: "`StandardizedTransformer(repo_id, ...)` picks the family from the config's `model_type`, renames the tree to the standard vocabulary, and takes every `TransformersModel` argument."
tags: [usage, loading, families, rename, envoys, tokenizer]
related: [docs/usage/vocabulary.md, docs/usage/availability.md, docs/usage/root-values.md, docs/usage/residual-stream.md]
sources: [nnterp/standardized.py, nnterp/families/__init__.py, nnterp/components/standard.py]
---

# Loading a model

## What this is for

`StandardizedTransformer` is an nnsight `TransformersModel` whose envoy tree also answers
to one set of names (`model.layers[i].self_attn`, `model.norm`, ...) and carries the
standard values (`layer_output`, `attention_output`, `logits`, ...). Loading one is the
same call as loading a `TransformersModel`, plus three keyword arguments of its own:
`rename=`, `envoys=` and `tokenizer_kwargs=`. This page is what happens at load and which
arguments matter for what you can read afterwards.

## Canonical pattern

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer(
    "openai-community/gpt2",
    dispatch=True,                  # load the weights now rather than at the first trace
    attn_implementation="eager",    # needed for the attention interior (pattern, scores, q/k/v)
    dtype=torch.bfloat16,
    device_map="auto",
)

print(model.family)                 # <module 'nnterp.families.gpt2' ...>
print(model.num_layers, model.hidden_size, model.num_heads)

with model.trace("The Eiffel Tower is in"):
    resid = model.layers[3].layer_output.save()
    pattern = model.layers[3].self_attn.attention_probabilities.save()
    logits = model.logits.save()
```

Every argument other than `rename`, `envoys` and `tokenizer_kwargs` goes straight to
`TransformersModel`: `dispatch`, `dtype`, `device_map`, `attn_implementation`, `revision`,
`trust_remote_code`, `task` and the rest. `task` defaults to `"text-generation"`.

## How the family is chosen

The constructor reads the checkpoint's config before any model is built, with
`AutoConfig.from_pretrained(repo_id, revision=..., trust_remote_code=...)`, so those two
arguments flow into the config read as well as the load. The config's `model_type` is the
family: `nnterp.families.<model_type>` is imported on first use (`gpt2.py` for `gpt2`,
`gemma3_text.py` for `gemma3_text`), and that module's `RENAME` and `ENVOYS` become
nnsight's `rename=` and `envoys=`. A multimodal config nests the language model's config
as `text_config`; the text-generation task builds that model, so its `model_type` is the
one looked up.

A `model_type` with no family raises `UnsupportedFamily` before anything loads, naming every
known model type, the list `nnterp.families.known()` returns (92 shipped families,
alphabetical):

```
UnsupportedFamily: no standardization for model_type 'bert'; known: ['afmoe', 'apertus', 'arcee',
'bamba', ..., 'xglm', 'youtu', 'zaya']. Add nnterp/families/bert.py with MODEL_TYPES, RENAME and
ENVOYS, or pass a module to nnterp.families.register().
```

`model.family` is the module the checkpoint resolved to; `nnterp.families.known()` lists
the shipped ones without loading anything.

## Passing an already-loaded module

`repo_id` can be a `torch.nn.Module` instead of a string. Its own `config` is read for the
family and the module is wrapped as is, so a model you built or edited yourself gets the
same tree:

```python
from transformers import AutoModelForCausalLM

raw = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", attn_implementation="eager")
model = StandardizedTransformer(raw)
model.family.__name__            # 'nnterp.families.gpt2'

with model.trace("Hello world"):
    x = model.layers[0].layer_output.save()
```

## `device` and `device_map`

`device="cpu"` (or `"cuda:0"`) puts the whole model on that device. `device_map` is passed on
to transformers, but nnsight's pipeline passes its own `device` too, which wins: a model loaded
with `device_map="cpu"` on a machine with a GPU lands on `cuda:0`. Use `device=` to choose
one device, and `device_map="auto"` to spread a model over several:

```python
model = StandardizedTransformer("openai-community/gpt2", dispatch=True, device="cpu")
next(model._module.parameters()).device          # device(type='cpu')
```

## `dispatch` and `attn_implementation`

Without `dispatch=True` the model is built on the meta device and the weights load at the
first trace, as with any `TransformersModel`; `model.support()` and the sizes work either
way, since they read the config.

`attn_implementation` is transformers' argument and it keeps transformers' default
(`sdpa` where available). The attention interior (`attention_probabilities`,
`attention_scores`, `attention_queries`, `attention_keys`, `attention_values`,
`attention_head_outputs`) is read inside the eager attention forward, so under any other
implementation those values are unavailable and `support()` says so:

```
"read inside the eager attention forward, but this model runs 'sdpa'; load with attn_implementation='eager'"
```

The boundary values (`layer_output`, `attention_output`, `mlp_output`, the root's `logits`,
`token_embeddings`, `next_token_probs`, `input_ids`) do not depend on it. Pass
`attn_implementation="eager"` whenever the interior is on the menu; see
[availability](availability.md).

## Your own `rename=` and `envoys=`

Both merge over the family's, and a key you give wins:

```python
model = StandardizedTransformer(
    "openai-community/gpt2",
    dispatch=True,
    rename={"transformer.wpe": "embed_positions", "mlp": "ffn"},
)
model.embed_positions is model.transformer.wpe     # True: a dotted key mounts on the root
model.layers[0].ffn is model.layers[0].mlp         # True: a single-component key binds in every block
model.layers[0].self_attn is model.transformer.h[0].attn   # the family's aliases are still there
```

The key forms are nnsight's (nnsight docs/usage/rename-modules.md): a dotted key binds
where it resolves from, the root; a single-component key binds on every envoy that has a
child of that name; a key that resolves nowhere is skipped.

`envoys=` maps a module *type* or a *native* dotted path to an `Envoy` subclass. nnsight
tries the type keys first, over the module's MRO, then the path keys, so:

```python
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP
from nnterp.families import gpt2

class MyMlp(gpt2.Mlp):
    ...

# replaces the family's Mlp on every GPT2MLP: key on the same type
model = StandardizedTransformer("openai-community/gpt2", dispatch=True, envoys={GPT2MLP: MyMlp})
type(model.layers[0].mlp) is MyMlp                 # True
```

A path key reaches only a module no type key in the merged map matches. The family keys
its block, attention and MLP by type, so `envoys={"transformer.h.0.mlp": MyMlp}` leaves
the family's `Mlp` in place there, while `envoys={"ln_f": MyNorm}` wraps the final norm,
which no family keys. Paths are the native ones (`transformer.h.0.mlp`); an alias
(`layers.0.mlp`) matches nothing, because aliases are bound after the envoys are chosen.
When the load shards across GPUs (`distributed_config=`), nnsight's tensor-parallel envoys
are the base of the merge, so a family's or your own map never drops them.

## The tokenizer

`tokenizer_kwargs=` sets attributes on the loaded tokenizer, for the two a batch of
prompts usually needs:

```python
model = StandardizedTransformer(
    "meta-llama/Llama-3.1-8B",
    tokenizer_kwargs={"padding_side": "left", "pad_token": "<|end_of_text|>"},
)
model.tokenizer.padding_side       # 'left'
```

`model.add_prefix_false_tokenizer` is the checkpoint's tokenizer loaded with
`add_prefix_space=False`, once, on first use, so `"word"` and `" word"` tokenize to
different first tokens. `nnterp.prompt_utils.get_first_tokens` uses it.

## What the repr shows

Standard names print beside the native name as `alias/native`, and every standard value
prints with its layout (the alias name and its axes, when the value has one) and its description, so `print(model.layers[i])` is the reference for what a block
of this family has:

```
GPT2Block(
  (input_layernorm/ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True, bias=True)
  (self_attn/attn): GPT2Attention(
    (c_attn): Conv1D()
    (c_proj): Conv1D()
    (attn_dropout): Dropout(p=0.1, inplace=False)
    (resid_dropout): Dropout(p=0.1, inplace=False)
    (attention_queries) -> Queries [batch heads seq qk_head_dim]: The queries entering attention
    (attention_keys) -> Keys [batch kv_heads seq qk_head_dim]: The keys entering attention
    (attention_values) -> Values [batch kv_heads seq head_dim]: The values entering attention
    (attention_scores) -> Pattern [batch heads query key]: The attention scores entering the softmax, masked
    (attention_output) -> Residual [batch seq hidden]: What the attention adds to the residual stream
    (attention_probabilities) -> Pattern [batch heads query key]: The attention pattern the values are mixed with
    (attention_head_outputs) -> HeadOutputs [batch seq heads head_dim]: The per-head outputs before the output projection
  )
  (post_attention_layernorm/ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True, bias=True)
  (mlp): GPT2MLP(
    (c_fc): Conv1D()
    (c_proj): Conv1D()
    (act): NewGELUActivation()
    (dropout): Dropout(p=0.1, inplace=False)
    (mlp_output) -> Residual [batch seq hidden]: What the MLP adds to the residual stream
  )
  (layer_output) -> Residual [batch seq hidden]: The residual stream leaving the block, a tensor even when the block returns a tuple
)
```

The root's own values (`logits`, `token_embeddings`, `next_token_probs`, `input_ids`,
`attention_mask`, `input_size`) print the same way at the end of `print(model)`; see
[root-values](root-values.md). A value is listed whether or not this checkpoint has it;
`support()` is what says.

## Gotchas

- **Import `nnterp` (or `nnsight`) before any `transformers.models...` or
  `transformers.modeling_layers` import.** The reverse order segfaults at import on this
  stack. A plain `import transformers` first is fine.
- **`attn_implementation` is not defaulted to eager.** A model loaded without it runs
  transformers' default, and the attention interior is unavailable; the boundary values
  still work.
- **Your `envoys=` key must be the type to displace a family envoy.** Type keys are tried
  before path keys, and the family keys its modules by type.
- **`envoys=` paths are native, never aliases.** `"layers.0.mlp"` matches nothing on GPT-2;
  `"transformer.h.0.mlp"` is the path, and even that is shadowed by the family's type key.
- **A `rename=` alias that would shadow an existing name raises at construction** (an
  `Envoy` attribute, a sibling module, a name on the wrapped model). That is nnsight's rule;
  see nnsight docs/usage/rename-modules.md.
- **The config is read twice** (once for the family, once by the load); with
  `HF_HUB_OFFLINE=1` both reads hit the cache.
- **`device_map="cpu"` does not keep a model off the GPU**; `device="cpu"` does.

## Related

- [vocabulary](vocabulary.md): the standard names and how they map per family.
- [availability](availability.md): `support()` and what a checkpoint lacks.
- [root-values](root-values.md): `logits`, `token_embeddings`, sizes.
- [residual-stream](residual-stream.md): the three contribution values.
- nnsight docs/usage/rename-modules.md: the `rename=` key forms.
