# Coming from nnterp 1.x

nnterp 1.x reached a block's tensors through accessors on the model, `model.layers_output[i]` and `model.attentions_output[i]`; nnterp 2 puts them on the block's envoy, `model.layers[i].layer_output` and `model.layers[i].self_attn.attention_output`, and the constructor validates nothing.

| nnterp 1.x | nnterp 2 | status |
| --- | --- | --- |
| `model.embed_tokens`, `model.layers[i]`, `model.lm_head` | same | same |
| `model.ln_final` | `model.norm` | renamed |
| `model.layers_input[i]` | `model.layers[i].input` | renamed |
| `model.layers_output[i]` | `model.layers[i].layer_output` | renamed |
| `model.attentions[i]`, `model.attentions_input[i]` | `model.layers[i].self_attn`, `model.layers[i].self_attn.input` | renamed |
| `model.attentions_output[i]` | `model.layers[i].self_attn.attention_output` | changed: the contribution, not the module's output |
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

## Accessors are on the blocks

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

## The attention pattern

1.x code passed `enable_attention_probs=True`, which forced the eager attention and checked the pattern at load, and asked `model.attn_probs_available`. nnterp 2 forces nothing: pass `attn_implementation="eager"` yourself and ask `support()`, which answers from the config, before any trace. Reading the pattern on a model that runs `sdpa` raises `nnterp.Unavailable` at that line.

!!! warning "The 1.x flags are accepted and ignored"
    `enable_attention_probs=True`, `check_renaming=`, `allow_dispatch=` and the other 1.x keywords pass through to `TransformersModel` and have no effect, with no error and no warning. A load with `enable_attention_probs=True` runs `sdpa`, and the pattern is unavailable.

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

`support()` returns `None` for a value every block has, else the reason, per block. Ask it instead of `hasattr`: `hasattr(envoy, "attention_probabilities")` raises `Unavailable` when the value is missing rather than returning `False`.

## Nothing runs at construction

1.x ran a renaming check and a dummy forward at load and raised `RenamingError` when a module was not where it expected. nnterp 2 reads the config's `model_type`, picks the family and builds the tree; a shipped family is used as it is, a `model_type` with no family loads under a best-effort default with a warning, and `UnsupportedFamily` is raised only when that default cannot name the modules either. A wrong assumption fails at the first trace instead, as `Unavailable` or nnsight's `SourceNotAvailable`, so run `model.support()` and one trace of the values you need before a long experiment. `remote=True` belongs on `trace()`, where nnsight puts it, and the constructor no longer sets `device_map="auto"` for you.

## `steer` and `skip_layers`

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

## `project_on_vocab` on Gemma-2

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

## Helpers and errors

`nnterp.nnsight_utils` keeps `get_token_activations`, the two `collect_*` functions and `compute_next_token_probs` with their 1.x signatures; the `get_layer_output` style accessors are gone, and `model.layers[i].layer_output` is what they wrapped. `nnterp.prompt_utils` is unchanged except that `run_prompts` shows no progress bar unless you pass `tqdm=`. `nnterp.interventions` and `nnterp.display` have no counterpart: `logit_lens` is a loop over `model.project_on_vocab(layer.layer_output)`, which the [Logit lens](../tutorials/logit-lens.md) tutorial writes out. `RenamingError` is replaced by `nnterp.Unavailable` (a value this checkpoint lacks, with the reason) and `nnterp.UnsupportedFamily` (a `model_type` nothing can standardize).
