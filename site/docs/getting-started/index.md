# Getting Started

nnterp gives every transformer family the same module names and the same standard values, on top of [nnsight](https://nnsight.net). This page installs it, runs a first trace, and sets that trace beside its nnsight form and its nnterp 1.x form. For the model you are about to load, the [encyclopedia](https://ndif-team.github.io/nnterp/encyclopedia/) has a page per supported family: the block drawn with nnterp's names and every value on it.

## Installation

nnterp is a PyPI package for Python 3.10 or later. It depends on nnsight 0.8 and transformers 5.17 or later, which pip installs with it.

```bash
pip install nnterp
```

### From source

```bash
pip install git+https://github.com/ndif-team/nnterp.git
```

For a checkout that you intend to edit, run `pip install -e .` from the clone.

### Extras

```bash
pip install "nnterp[models]"   # sentencepiece, tiktoken, protobuf, pillow: tokenizers and processors some checkpoints need
pip install "nnterp[vllm]"     # vLLM, for StandardizedVLLM
pip install "nnterp[test]"     # pytest and pytest-xdist, for the test suite
```

### For agents

If an agent writes your nnterp code, give it the nnterp skills from [ndif-team/skills](https://github.com/ndif-team/skills): three skills, for the values, for the recipes that run across families, and for extending nnterp. In Claude Code:

```
/plugin marketplace add https://github.com/ndif-team/skills.git
/plugin install nnterp@ndif-team
```

In Codex, `$skill-installer install https://github.com/ndif-team/skills`. Every page of this site is also served as Markdown, listed in [llms.txt](../llms.txt), with the whole site in [llms-full.txt](https://ndif-team.github.io/nnterp/llms-full.txt).

## Quick start

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

### The names

`model.embed_tokens`, `model.layers[i]`, `model.layers[i].self_attn`, `model.layers[i].mlp`, `model.norm` and `model.lm_head` exist on every family. Because they are nnsight aliases, the native names keep working and both reach the same envoy:

```python
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2")

print(model.layers[5] is model.transformer.h[5], model.layers[5].self_attn is model.transformer.h[5].attn)
# True True
```

### The values

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

### Availability

Not every checkpoint has every value. `model.support()` says, from the config and the module tree, what this one has and, for anything missing, why. A family's [encyclopedia](https://ndif-team.github.io/nnterp/encyclopedia/) page shows the same answer for each of its public checkpoints. Reading a missing value inside a trace raises `nnterp.Unavailable` with the same reason, before the model runs:

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

### Methods

The operations that an experiment repeats are methods on the model, written against the values, so they run on every family:

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

## Coming from nnsight

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

The same trace with nnterp's names and values is the first block of the [quick start](#quick-start): the four tensors are the ones that the nnsight trace saved (the next block checks one), and that body runs unchanged on `EleutherAI/pythia-70m-deduped`, where the attention is `gpt_neox.layers[5].attention`, and on `bigscience/bloom-560m`, where the block returns a tuple and `layer_output` is still the tensor.

### What does not change

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

## Coming from nnterp 1.x

nnterp 1.x reached a block's tensors through accessors on the model, `model.layers_output[i]` and `model.attentions_output[i]`; nnterp 2 puts them on the block's envoy, `model.layers[i].layer_output` and `model.layers[i].self_attn.attention_output`, and the constructor validates nothing.

| nnterp 1.x | nnterp 2 | status |
| --- | --- | --- |
| `model.embed_tokens`, `model.layers[i]`, `model.lm_head` | same | same |
| `model.ln_final` | `model.norm` | renamed |
| `model.layers_input[i]` | `model.layers[i].input` | renamed |
| `model.layers_output[i]` | `model.layers[i].layer_output` | renamed |
| `model.attentions[i]`, `model.attentions_input[i]` | `model.layers[i].self_attn`, `model.layers[i].self_attn.input` | renamed |
| `model.attentions_output[i]` | `model.layers[i].self_attn.attention_output` | changed: the contribution, not the module's output; the same tensor on GPT-2 and Llama, different only where the family norms the module's output (Gemma-2) or adds the residual inside it (BLOOM) |
| `model.mlps[i]`, `model.mlps_input[i]` | `model.layers[i].mlp`, `model.layers[i].mlp.input` | renamed |
| `model.mlps_output[i]` | `model.layers[i].mlp.mlp_output` | changed: the contribution, not the module's output |
| `model.attention_probabilities[i]` | `model.layers[i].self_attn.attention_probabilities` | renamed |
| `model.logits`, `token_embeddings`, `next_token_probs`, `input_ids`, `input_size`, `attention_mask` | same; `logits`, `input_ids` and `attention_mask` are also assignable | same |
| `model.num_layers`, `num_heads`, `hidden_size`, `vocab_size` | same, read-only | same |
| `model.attn_probs_available` | `model.support()["self_attn.attention_probabilities"] is None` | changed |
| `StandardizedTransformer(repo, enable_attention_probs=True)` | `StandardizedTransformer(repo, attn_implementation="eager")` | changed |
| `check_renaming=`, `allow_dispatch=`, `check_attn_probs_with_trace=`, `allow_multimodal=` | nothing runs at construction | gone |
| `rename_config=RenameConfig(...)` | `rename=`, `envoys=`, `family=`, `nnterp.families.register` | changed |
| `StandardizedTransformer(repo, remote=True)` | `model.trace(prompt, remote=True)` | changed |
| `device_map="auto"` unless given | nnsight's default; pass `device_map="auto"` | changed |
| `trust_remote_code=`, `tokenizer_kwargs=`, `rename=` | same | same |
| `model.skip_layer(i)` | `model.skip_layers(i, i)` | gone |
| `model.skip_layers(start, end, skip_with, layer_returns_tuple)`, `detect_layer_output_type()` | `model.skip_layers(start, end, skip_with)` | changed |
| `model.steer(layers, steering_vector, factor, positions, token_positions, batch_index)` | `model.steer(layers, vector, factor, token_positions, batch_index)` | changed |
| `model.project_on_vocab(h)` | same name; applies the softcap on Gemma-2 | changed |
| `model.probs_to_dict(tokens, probs)` | `model.probs_to_dict(probs, k=5)` | changed |
| `model.get_topk_closest_tokens(h, k)`, a dict for a 1-D `h` | always a list of dicts | changed |
| `nnterp.nnsight_utils.get_layer_output(model, i)` and the other `get_*`, `ModuleAccessor` | none: the names and values above | gone |
| `get_token_activations`, `collect_token_activations_batched`, `collect_last_token_activations_session`, `compute_next_token_probs` | same | same |
| `nnterp.prompt_utils`: `Prompt`, `get_first_tokens`, `run_prompts` | same; `run_prompts` shows a progress bar only when given `tqdm=` | same |
| `nnterp.interventions`: `logit_lens`, `patchscope_lens`, `patchscope_generate`, `repeat_prompt`, `TargetPrompt` | none; the [Logit lens](../tutorials/logit-lens.md) and [Activation patching](../tutorials/activation-patching.md) tutorials | gone |
| `nnterp.display.plot_topk_tokens`, `prompts_to_df` | none | gone |
| `RenamingError`, `get_rename_dict` | `nnterp.Unavailable`, `nnterp.UnsupportedFamily` | changed |

### `layers_output` and the other accessors

1.x code read `model.layers_output[5]` and wrote `model.layers_output[5] = v`; nnterp 2 code reads and writes `model.layers[5].layer_output`. The module accessors `attentions[i]` and `mlps[i]` are the envoys `layers[i].self_attn` and `layers[i].mlp`, and `ln_final` is `norm`:

```python
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2")

with model.trace("The Eiffel Tower is in"):
    x = model.layers[5].input.save()                          # layers_input[5]
    attn = model.layers[5].self_attn.attention_output.save()  # attentions_output[5]
    mlp = model.layers[5].mlp.mlp_output.save()               # mlps_output[5]
    model.layers[5].layer_output[:, -1] += 0.5 * x[:, -1]     # layers_output[5] edited in place
    normed = model.norm.output.save()                         # ln_final_output
    logits = model.logits.save()

print(x.shape, attn.shape, mlp.shape, normed.shape, logits.shape)
# torch.Size([1, 7, 768]) torch.Size([1, 7, 768]) torch.Size([1, 7, 768]) torch.Size([1, 7, 768]) torch.Size([1, 7, 50257])
```

`attentions_output[i]` and `mlps_output[i]` were the module's output, first element of a tuple; `attention_output` and `mlp_output` are what the sublayer adds to the residual stream, and `layers[i].input + attention_output + mlp_output == layer_output` on every family. On GPT-2 and Llama the two are the same tensor. On BLOOM the attention module returns the stream with the residual already added, and on Gemma-2 the block adds a post-norm's output, so 1.x code that read the module's output there now reads a different tensor:

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("bigscience/bloom-560m")

with model.trace("The Eiffel Tower is in"):
    x = model.layers[2].input.save()
    attn = model.layers[2].self_attn.attention_output.save()   # the contribution
    raw = model.layers[2].self_attn.output[0].save()           # what 1.x attentions_output[2] gave

print(torch.allclose(raw, x + attn))
# True
```

### Read order

nnterp 2 reads values in the model's forward order within a trace and raises `OutOfOrderError` otherwise. 1.x code that read block 5 before block 3 in one trace has to be reordered ([Attention: read order](../guide/values/attention.md#read-order)).

### The attention pattern

1.x code passed `enable_attention_probs=True`, which forced the eager attention and checked the pattern at load, and asked `model.attn_probs_available`. nnterp 2 forces nothing: pass `attn_implementation="eager"` yourself and ask `support()`, which answers from the config, before any trace. Reading the pattern on a model that runs `sdpa` raises `nnterp.Unavailable` at that line.

!!! warning "The 1.x flags reach transformers, which rejects them"
    `enable_attention_probs=True`, `check_renaming=`, `allow_dispatch=` and the other 1.x keywords pass through `TransformersModel` to the checkpoint's constructor, which raises `TypeError: GPT2LMHeadModel.__init__() got an unexpected keyword argument 'enable_attention_probs'`, inside a `ValueError` from the loader. With `dispatch=True` that happens at load. Without it the load reads only the config, `support()` reports `sdpa`, and the error comes at the first trace. Remove the keywords.

```python
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", attn_implementation="eager")
print(model.support()["self_attn.attention_probabilities"])
# None

with model.trace("The Eiffel Tower is in"):
    pattern = model.layers[5].self_attn.attention_probabilities.save()   # attention_probabilities[5]

print(pattern.shape, pattern[0, 0].sum(-1))
# torch.Size([1, 12, 7, 7]) tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
#        grad_fn=<SumBackward1>)
```

`support()` returns `None` for a value that every block has, else the reason, per block. Ask it instead of `hasattr`: `hasattr(envoy, "attention_probabilities")` raises `Unavailable` when the value is missing rather than returning `False`.

### The constructor

1.x ran a renaming check and a dummy forward at load and raised `RenamingError` when a module was not where it expected. nnterp 2 reads the config's `model_type`, picks the family and builds the tree. A shipped family is used as it is. A `model_type` with no family loads under a best-effort default with a warning, and `UnsupportedFamily` is raised only when that default cannot name the modules either. A wrong assumption fails at the first trace instead, as `Unavailable` or nnsight's `SourceNotAvailable`, so run `model.support()` and one trace of the values that you need before a long experiment. `remote=True` belongs on `trace()`, where nnsight puts it, and the constructor no longer sets `device_map="auto"` for you. To keep a model on the CPU pass `device="cpu"`, since `device_map="cpu"` does not ([Loading](../guide/loading.md#device-versus-device_map)).

### `steer` and `skip_layers`

The steering vector is `vector`, positions are `token_positions` only, and a list of `layers` has to be ascending because nothing sorts it. `skip_layer(i)` is `skip_layers(i, i)`, and `layer_returns_tuple` is gone, since each family declares whether its blocks return a tuple:

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2")
torch.manual_seed(0)
vector = torch.randn(model.hidden_size)

with model.trace("The Eiffel Tower is in"):
    model.skip_layers(3, 3)                                       # skip_layer(3)
    model.steer([4, 6], vector, factor=2.0, token_positions=-1)   # steer([4, 6], steering_vector=vector, factor=2.0, positions=-1)
    resid = model.layers[6].layer_output.save()

print(model.get_topk_closest_tokens(resid[0, -1], k=3))
# [{' FIRE': 0.4005129635334015, ' exchange': 0.06274925172328949, ' smart': 0.0324697308242321}]
```

`get_topk_closest_tokens` returns a list of one dict per position whatever the rank of its input, where 1.x returned a bare dict for a single position, and its keys are decoded text (`' FIRE'`) rather than vocabulary tokens (`'ĠFIRE'`). `probs_to_dict` takes a distribution and `k`, not the token ids and their probabilities.

### `project_on_vocab` on Gemma-2

1.x computed `lm_head(ln_final(h))`. nnterp 2 adds what the model does after the head, which on Gemma-2 is the softcap `final_logit_softcapping * tanh(logits / final_logit_softcapping)`: a logit lens at the last block now equals `logits`, and the 1.x lens on Gemma-2 did not. On GPT-2, Llama and every family with nothing after the head the two agree.

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("google/gemma-2-2b")

with model.trace("The Eiffel Tower is in"):
    last = model.layers[-1].layer_output.save()
    logits = model.logits.save()

lens = model.project_on_vocab(last)
raw = model.lm_head(model.norm(last))            # what 1.x computed
print(torch.equal(lens, logits), torch.equal(raw, logits), (raw - logits).abs().max().item())
# True False 51.002906799316406
```

### Helpers and errors

`nnterp.nnsight_utils` keeps `get_token_activations`, the two `collect_*` functions and `compute_next_token_probs` with their 1.x signatures. The `get_layer_output` style accessors are gone, and `model.layers[i].layer_output` is what they wrapped. `nnterp.prompt_utils` is unchanged except that `run_prompts` shows no progress bar unless you pass `tqdm=`. `nnterp.interventions` and `nnterp.display` have no counterpart: `logit_lens` is a loop over `model.project_on_vocab(layer.layer_output)`, which the [Logit lens](../tutorials/logit-lens.md) tutorial writes out. `RenamingError` is replaced by `nnterp.Unavailable` (a value that this checkpoint lacks, with the reason) and `nnterp.UnsupportedFamily` (a `model_type` that nothing can standardize).

The [Guide](../guide/index.md) has one page per name and value.
