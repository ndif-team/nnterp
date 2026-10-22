# Loading

`StandardizedTransformer(repo_id, ...)` is nnsight's `TransformersModel` with one step before the load: it reads the checkpoint's config, takes the family from `model_type`, and hands that family's aliases and envoy classes to nnsight. Every `TransformersModel` keyword passes through unchanged.

```python
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", dispatch=True, device="cpu")
print(model.config.model_type, model.family.__name__)
print(model.num_layers, model.hidden_size, model.num_heads)

with model.trace("The Eiffel Tower is in"):
    resid = model.layers[3].layer_output.save()
    logits = model.logits.save()
print(resid.shape, logits.shape)
# gpt2 nnterp.families.gpt2
# 12 768 12
# torch.Size([1, 7, 768]) torch.Size([1, 7, 50257])
```

`model.family` is the module `nnterp.families.<model_type>`, imported on first use, and `nnterp.families.known()` lists the shipped ones. A multimodal config nests the language model's config as `text_config`, and that inner `model_type` is the one looked up, so a Gemma 3 wrapper resolves to `gemma3_text` under either task.

## Arguments

`family=`, `rename=`, `envoys=` and `tokenizer_kwargs=` are nnterp's own. Everything else, such as `dispatch`, `dtype`, `device_map`, `attn_implementation`, `revision` and `task`, goes to `TransformersModel` as documented at [nnsight.net](https://nnsight.net/features/5_loading/), and `revision` and `trust_remote_code` also reach the config read that picks the family. The full signature is under the Documentation tab ([standardized](../documentation/standardized.md)). `task` defaults to `"text-generation"`, and `task="image-text-to-text"` loads a vision-language checkpoint as its wrapper, with its processor ([Vision-language](architectures/vision-language.md)). Without `dtype` the weights load in the config's own dtype, float32 for GPT-2 and bfloat16 for Qwen2.5. Without `dispatch=True` the model is built on the meta device and the weights load at the first trace, while `model.support()` and the sizes, which read the config, work either way.

### `attn_implementation="eager"`

The values inside attention (`attention_probabilities`, `attention_scores`, the queries, keys, values and head outputs) are read inside transformers' eager attention forward. The default implementation is the checkpoint's, usually `sdpa`, and those values are then unavailable:

```python
model = StandardizedTransformer("openai-community/gpt2", dispatch=True, device="cpu")
print(model.config._attn_implementation)
print(model.support(layer=0)["self_attn.attention_probabilities"])
# sdpa
# read inside the eager attention forward, but this model runs 'sdpa'; load with attn_implementation='eager'
```

Pass `attn_implementation="eager"` whenever a trace will touch the inside of attention. The boundary values (`layer_output`, `attention_output`, `mlp_output`, `logits`) do not depend on it ([Availability](values/availability.md)).

### `device=` versus `device_map=`

`device="cpu"` keeps the whole model on the CPU. `device_map="cpu"` does not: nnsight's pipeline passes its own `device` as well, and on a machine with a GPU the model lands on `cuda:0`. Use `device=` for one device and `device_map="auto"` to spread a model over several.

```python
on_cpu = StandardizedTransformer("openai-community/gpt2", dispatch=True, device="cpu")
mapped = StandardizedTransformer("openai-community/gpt2", dispatch=True, device_map="cpu")
print(on_cpu.device, mapped.device)
# cpu cuda:0
```

### `rename=` and `envoys=`

Both merge over the family's, and a key you give wins. The key forms are nnsight's ([Module Access](https://nnsight.net/features/6_modules/)): a dotted key resolves from the root and mounts there, and a single-component key binds on every envoy that has a child of that name.

```python
model = StandardizedTransformer(
    "openai-community/gpt2", dispatch=True, device="cpu",
    rename={"transformer.wpe": "embed_positions", "mlp": "ffn"},
)
print(model.embed_positions is model.transformer.wpe)
print(model.layers[0].ffn is model.layers[0].mlp)
print(model.layers[0].self_attn is model.transformer.h[0].attn)
# True
# True
# True
```

`envoys=` maps a module type or a dotted path to an `Envoy` subclass. nnsight tries the type keys first, and the families key their envoys by type, so only a type key displaces a family's envoy. With `envoys={GPT2MLP: MyMlp}`, a subclass of `nnterp.families.gpt2.Mlp`, `type(model.layers[0].mlp) is MyMlp` on every block. A path key reaches a module that no type key in the merged map matches, by its native path or any alias spelling of it (`"norm"` reaches GPT-2's `ln_f`).

### `tokenizer_kwargs=`

`tokenizer_kwargs=` sets attributes on the loaded tokenizer, for the two that a batch of prompts usually needs.

```python
model = StandardizedTransformer(
    "openai-community/gpt2", dispatch=True, device="cpu",
    tokenizer_kwargs={"padding_side": "left", "pad_token": "<|endoftext|>"},
)
with model.trace(["Hello", "The Eiffel Tower is in"]):
    mask = model.attention_mask.save()
print(mask)
# tensor([[0, 0, 0, 0, 0, 0, 1],
#         [1, 1, 1, 1, 1, 1, 1]])
```

### `family=`

`family=` gives one load its family without reading the config or the registry: a module, or any object with `RENAME` and `ENVOYS` such as `types.SimpleNamespace(**vars(gpt2))`. To cover a `model_type` for every load, register the family instead ([Registering](../developing/registering.md)).

## A `model_type` with no family

A `model_type` with no family module and nothing registered loads with `nnterp.families.default`, a best-effort family, and a warning. The default maps the spellings the shipped families use onto Llama's names and checks its guess at load on a shape-only scan. It serves `layer_output`, the root values, the sizes and the attention interior, and it refuses what a name cannot tell: `attention_output`, `mlp_output`, `layer_mid`, `mlp_norm_output`, a mixture's interior and `project_on_vocab`. GPT-2 has a family, so this block uses a tiny random checkpoint of a type that has none.

```python
model = StandardizedTransformer("hf-tiny-v2/tiny-random-DiffLlamaForCausalLM", dispatch=True, device="cpu")
# UserWarning: nnterp has no family for model_type 'diffllama'; the default family standardizes it as a
# best-effort guess. Check model.support() for what it found, and add nnterp/families/diffllama.py (or
# nnterp.families.register(family, 'diffllama')) for a standardization you can rely on.
print(model.family.__name__)
support = model.support()
print(support["layer_output"])
print(support["mlp.mlp_output"][0])
# nnterp.families.default
# None
# the default family cannot tell what this sublayer adds to the stream; add a family module (docs/extending/adding-a-family.md)
```

When the guess finds no embedding, blocks, final norm or head, the load raises `UnsupportedFamily` and suggests a `rename=` read off the module tree; passing it completes the default's names. [Adding a family](../developing/adding-a-family.md) is the standardization you can rely on, and the families that have it are the pages of the [encyclopedia](https://ndif-team.github.io/nnterp/encyclopedia/).

```python
from nnterp import UnsupportedFamily

try:
    StandardizedTransformer("hf-tiny-v2/tiny-random-CTRLLMHeadModel", dispatch=True, device="cpu")
except UnsupportedFamily as error:
    print(error)
# the default family cannot standardize CTRLLMHeadModel: found no embed_tokens, norm under any name it knows.
# Name the native modules with rename={'<native path>': '<standard name>'} at load; going by the module tree,
# perhaps rename={'transformer.w': 'embed_tokens', 'transformer.layernorm': 'norm'}, or add a family module
# named after the model_type (docs/extending/adding-a-family.md).

model = StandardizedTransformer(
    "hf-tiny-v2/tiny-random-CTRLLMHeadModel", dispatch=True, device="cpu",
    rename={"transformer.w": "embed_tokens", "transformer.layernorm": "norm"},
)
print(model.family.__name__, model.support(layer=0)["layer_output"])
# nnterp.families.default None
```

## An already-loaded module

`repo_id` can be a `torch.nn.Module`. Its own `config` picks the family and the module is wrapped as it is, so a model you built or edited yourself gets the same tree.

```python
from transformers import AutoModelForCausalLM

raw = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", attn_implementation="eager")
model = StandardizedTransformer(raw)
print(model.family.__name__)
with model.trace("Hello world"):
    x = model.layers[0].layer_output.save()
print(x.shape)
# nnterp.families.gpt2
# torch.Size([1, 2, 768])
```

The names the loaded tree answers to are on [Names](names.md).
