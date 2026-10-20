# Architecture

A checkpoint's `model_type` selects one family module, and `StandardizedTransformer.__init__` hands that module's `RENAME` to nnsight as `rename=` and its `ENVOYS` as `envoys=`. nnsight then builds a tree in which every native envoy also answers to its standard alias and every block, attention and MLP is the family's `Layer`, `Attention` or `Mlp` subclass, and that class's `EProperty` descriptors are the values. The root class adds the whole-model values and the sizes.

```python
import nnterp
from nnterp import StandardizedTransformer
from nnterp.families import gpt2

model = StandardizedTransformer("openai-community/gpt2", dispatch=True, attn_implementation="eager", device="cpu")

print(model.family is gpt2)                                      # True
print(model._aliases)                                            # {'embed_tokens': 'transformer.wte', 'layers': 'transformer.h', 'norm': 'transformer.ln_f'}
print(model.layers[0]._aliases)                                  # {'input_layernorm': 'ln_1', 'self_attn': 'attn', 'post_attention_layernorm': 'ln_2'}
print(model.layers[0].self_attn is model.transformer.h[0].attn)  # True
print(type(model.layers[0]) is gpt2.Layer, type(model.layers[0].self_attn) is gpt2.Attention)  # True True
print(sorted(gpt2.Attention.values()))
# ['attention_head_outputs', 'attention_keys', 'attention_output', 'attention_premix', 'attention_probabilities', 'attention_queries', 'attention_scores', 'attention_values']

with model.trace("The Eiffel Tower is in"):
    pattern = model.layers[2].self_attn.attention_probabilities.save()
    resid = model.layers[2].layer_output.save()
    logits = model.logits.save()
print(tuple(pattern.shape), tuple(resid.shape), tuple(logits.shape))  # (1, 12, 7, 7) (1, 7, 768) (1, 7, 50257)
```

An alias is the same envoy object, so `model.layers[0].self_attn` is the `gpt2.Attention` that nnsight built for `transformer.h.0.attn`, and the descriptors on that class serve values at that module's native location. The pattern is read inside block 2's attention and the stream as block 2 returns, which is the forward order.

## From `model_type` to a family

`StandardizedTransformer.__init__` in `nnterp/standardized.py` reads the config before anything is built. `_read_config` returns a ready `nn.Module`'s own `config`, else `AutoConfig.from_pretrained` with the call's `revision` and `trust_remote_code`, and a config transformers cannot parse fails there with transformers' own error. On a multimodal config the lookup takes `text_config.model_type`, because the text model is what the family standardizes.

`families.lookup(model_type)` in `nnterp/families/__init__.py` consults `REGISTRY` first and then imports `nnterp.families.<model_type>`. A `ModuleNotFoundError` naming exactly that module means there is no such family, and `lookup` warns and returns `nnterp.families.default`. Any other import error comes from inside a family module and is re-raised as it is. A family module imports its transformers modeling module at its top, which keeps `import nnterp` free of modeling code and makes the first lookup of a type load only that family's module. `tests/test_registry.py` checks this in a subprocess.

```python
import sys
from nnterp import families

print(len(families.known()), families.lookup("gpt2") is gpt2)               # 98 True
print(sorted(m for m in sys.modules if m.startswith("nnterp.families.")))  # ['nnterp.families.default', 'nnterp.families.gpt2']
```

`known()` lists the package's modules by file name with `default` left out, so a module's name is its `model_type`. `default` is imported with nnterp itself, since the root compares `family is default` after the build. `register(family, *model_types)` writes into `REGISTRY`, and `family=` on the constructor skips the lookup for one load ([Registering](../registering.md)). Another engine's families form a package under this one: `lookup("llama", engine="vllm")` imports `nnterp.families.vllm.llama`, and since there is no default there a type without a module raises `UnsupportedFamily`.

## The two maps nnsight receives

The constructor merges `rename={**family.RENAME, **rename}` and `envoys={**tp_envoys, **family.ENVOYS, **envoys}`, later keys winning, so a caller's entry replaces the family's on the same key. `_base_envoys` starts from nnsight's tensor-parallel envoys on a sharded load because `TransformersModel` only defaults to them: A map passed by nnterp would otherwise drop them. Both maps are nnsight features ([Module Access](https://nnsight.net/features/6_modules/)).

Every `rename` key is resolved from every envoy after that envoy's children exist, and an alias that resolves is bound as another attribute of the same envoy object and recorded in `_aliases`. A key that resolves nowhere is skipped. A multi-component key such as `"transformer.h"` resolves only from the root, which is what puts `layers` on `model` rather than on `model.transformer`, while a single name such as `"attn"` resolves in every block that has one. A dotted alias such as `"attn.c_proj": "attn.o_proj"` binds on the attention.

The envoy class is chosen per module at construction: nnsight tries the module type's MRO against the type keys of `envoys`, then the string keys as dotted path suffixes, in the native spelling or any alias spelling of the path. A type key always beats a path key, which is why `envoys={"self_attn": X}` does nothing on GPT-2 while the family's `GPT2Attention` key stands, and why a caller displaces a family envoy by keying on the same type. The default family is the one that keys on names (`"layers.*"`, `"self_attn"`, `"mlp"`), which works because its own `RENAME` has aliased them first.

## Which layer owns what

| layer | holds | never knows |
|---|---|---|
| `nnterp/families/<model_type>.py` | `RENAME`; the `Layer`, `Attention`, `Mlp` subclasses (and `Moe`, a `RecurrentMixer`, the vision classes where the family has them) that point a value at this family's operation or sibling module; `ENVOYS` keyed on transformers types; `def <size>(model)` for a root size this config spells its own way; `def project_on_vocab(model, hidden)` where the model does something of its own after the head | what a value means or how nnsight serves it |
| `nnterp/components/` | what each value means and how it is read and written (`EProperty` paths, `select`, transforms); the availability predicates; the shared operation names such as `INTERFACE` | any one family's module names or classes |
| `nnterp/standardized.py` | the root values and methods, the sizes, `support()`, the load path, the remote key | an operation name inside a forward; a family's config keys |

Gemma-2's `attention_output` is the post-attention norm's output: The family keys it `"../post_attention_layernorm.output"` on its `Attention`, and the component only knows how to walk such a path. BLOOM's contribution is the first argument of its `dropout_add` call, so the family names `"source.dropout_add_0.input"`. A family on transformers' shared path, such as `nnterp/families/llama.py`, overrides nothing and is three container keys, three empty subclasses and the type keys.

## The root

`Standardized`, in `nnterp/standardized.py`, is the engine-independent part: the sizes, `support()`, `skip_layers`, `steer`, `get_topk_closest_tokens` and `_read_config`. `StandardizedTransformer` mixes it over `TransformersModel`, and `StandardizedVLLM` over nnsight's `VLLM` ([vLLM](../../vllm/index.md)). The root values are `EProperty` descriptors like a block's. `logits` reads the model's output and picks `.logits`, `token_embeddings` is keyed `"embed_tokens.output"`, `final_norm_output` and `lm_head_output` read `norm` and `lm_head`, `next_token_probs` is derived and refuses assignment, and `input_ids`, `attention_mask` and `input_size` are keyed `"inputs"` and read the call's keyword arguments.

Each size is a `StandardizedProperty`. On read it looks for a function of its own name on `model.family` and calls it with the model, else runs the plain rule over `config.get_text_config()`. Assignment raises `AttributeError`, which leaves the family module as the only place a size is spelled otherwise. `project_on_vocab` is a `StandardizedCapability`, the same lookup for a method.

```python
print(model.intermediate_size, model.config.n_inner)   # 3072 None
```

GPT-2's config leaves `n_inner` unset, and `gpt2.py` defines `intermediate_size` as four times the hidden size. `support()` collects the root values' reasons, then `blocks_support(model.layers)`: every `Standard` child of a block is listed under its standard name, the set of names is the union over the blocks, and a value missing on some blocks reads as `{layer: reason}`. A `Standard` child of the root, such as a vision tower, adds its own rows under its alias ([Vision](vision.md)).

## The default family

`nnterp/families/default.py` serves a `model_type` with no module. Its `RENAME` is the union of the spellings the shipped families use for the containers, the embedding, the blocks, the final norm, the head and a block's modules, and its `ENVOYS` key on the standard names. After the build the root calls `default.check(model)`, which raises `UnsupportedFamily` when a required root module is missing, when no block is a `Layer`, or when a shape-only `scan` shows that the blocks do not pass a `[batch, seq, hidden_size]` stream. It serves `layer_output`, the root values, the sizes and the attention interior wherever the module calls the shared interface. `attention_output`, `mlp_output` and `project_on_vocab` are unavailable, since what a sublayer adds and what follows the head depend on code a name does not show.

How a value's path becomes a served location is on the next page, [Descriptors](descriptors.md).
