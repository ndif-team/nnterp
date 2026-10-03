# nnterp

nnterp gives every transformer family the same module names and the same standard values,
on top of [nnsight](https://github.com/ndif-team/nnsight). `StandardizedTransformer` is an
nnsight `TransformersModel`, so `trace`, `generate`, `.save()`, `tracer.iter` and
`remote=True` work as they do in nnsight. An experiment written once runs on GPT-2, Llama,
Qwen, Gemma, Mamba and the other supported families.

```python
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2")   # or a Llama, a Pythia, ...

with model.trace("The Eiffel Tower is in"):
    attn = model.layers[5].self_attn.attention_output.save()   # what attention adds
    resid = model.layers[5].layer_output.save()                # the residual stream leaving block 5
    logits = model.logits.save()

print(attn.shape, resid.shape, logits.shape)
# torch.Size([1, 7, 768]) torch.Size([1, 7, 768]) torch.Size([1, 7, 50257])
```

## The idea

**One vocabulary.** Every family answers to Llama's names, lifted out of the inner `.model`:
`model.embed_tokens`, `model.layers[i]`, `model.layers[i].self_attn`, `model.layers[i].mlp`,
`model.norm`, `model.lm_head`. They are nnsight aliases, so GPT-2's `model.transformer.h[5]`
and `model.layers[5]` are the same envoy and the native names keep working.

**Standard values.** A name says which module; a value says which tensor, and it means the
same thing on every family. `layer_output` is the residual stream leaving a block, whether
the block returns a tensor or a tuple. `attention_output` and `mlp_output` are what each
sublayer adds to the residual stream, wherever the family adds the residual or applies a
post-sublayer norm, so on every family

```
layers[i].input + attention_output + mlp_output == layers[i].layer_output
```

Values can be read, edited in place, or assigned.

**Ask before you trace.** Not every checkpoint has every value. `model.support()` says what
this one has and, for anything missing, why; reading a missing value raises
`nnterp.Unavailable` with the same reason.

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
print(model.support(layer=5)["self_attn.attention_probabilities"])
# read inside the eager attention forward, but this model runs 'sdpa'; load with attn_implementation='eager'
```

## What you get

| area | values and methods | docs |
| --- | --- | --- |
| residual stream | `layer_output`, `attention_output`, `mlp_output`, `self_attn.input`, `mlp.input` | [residual-stream](docs/usage/residual-stream.md) |
| attention interior | `attention_probabilities`, `attention_queries` / `keys` / `values`, `attention_scores`, `attention_head_outputs` (load with `attn_implementation="eager"`) | [attention-interior](docs/usage/attention-interior.md) |
| mixture of experts | `router_logits`, `expert_weights`, `expert_indices`, `expert_outputs`, `routed_output`, `shared_expert_output` | [mixture-of-experts](docs/usage/mixture-of-experts.md) |
| recurrent mixers and hybrids | `linear_attn` on gated DeltaNet, Mamba and Mamba-2 blocks: queries, keys, values, `decays`, `betas`, the recurrent state | [delta-net](docs/usage/delta-net.md), [selective-scan](docs/usage/selective-scan.md), [state-space](docs/usage/state-space.md) |
| the whole model | `logits`, `token_embeddings`, `next_token_probs`, `input_ids`, and the sizes (`num_layers`, `hidden_size`, `head_dim`, ...) | [root-values](docs/usage/root-values.md) |
| methods | `steer`, `skip_layer`, `skip_layers`, `project_on_vocab`, `get_topk_closest_tokens` | [methods](docs/usage/methods.md) |
| interventions | `logit_lens`, `patchscope_lens`, `patchscope_generate`, `patch_object_attn_lens`, `TargetPrompt`, `repeat_prompt` | [interventions](docs/usage/interventions.md) |

Every value has one axis layout on every family, named in `nnterp.components`
([layouts](docs/usage/layouts.md)).

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", attn_implementation="eager")
vector = torch.randn(model.hidden_size)

with model.trace("The Eiffel Tower is in"):
    pattern = model.layers[3].self_attn.attention_probabilities.save()  # [batch, heads, query, key]
    lens = model.project_on_vocab(model.layers[6].layer_output).save()  # logit lens at block 6
    model.steer(8, vector, factor=3, token_positions=-1)                # add to block 8's output
    model.skip_layers(10, 11)                                           # blocks 10..11 do not run
    logits = model.logits.save()
```

## Supported families

92 families, among them GPT-2, Llama, Mistral, Qwen 2/3/3.5, Gemma 1-4, Phi, OLMo, GPT-NeoX,
DeepSeek-V2/V3, GPT-OSS, Mixtral, Falcon, BLOOM, Mamba, Jamba and Nemotron-H, developed
against transformers 5.17. The full table, with each family's native names and quirks, is
[docs/reference/families.md](docs/reference/families.md).

## Installation

```
pip install nnterp
```

Extras:

- `pip install "nnterp[models]"`: the tokenizers and processors some checkpoints need
  (`sentencepiece`, `tiktoken`, `protobuf`, `pillow`).
- `pip install "nnterp[vllm]"`: vLLM.
- `pip install "nnterp[test]"`: `pytest` and `pytest-xdist`, for the test suite.

nnterp requires nnsight 0.8.0 or later. PyPI has only the 0.8.0rc1 pre-release, which
`nnsight>=0.8` does not accept, so `pip install nnterp` resolves once nnsight 0.8.0 is published.

## Documentation

- [docs/usage/](docs/usage/index.md): one page per feature; start with
  [loading](docs/usage/loading.md) and [vocabulary](docs/usage/vocabulary.md).
- [docs/patterns/](docs/patterns/index.md): logit lens, steering, attention patterns,
  ablation, activation patching, probing and more, written once against the standard values.
- [docs/extending/](docs/extending/index.md): add a family, override a value, add your own value.
- [docs/reference/](docs/reference/index.md): the API quick reference, the families table, a glossary.
- [docs/developing/](docs/developing/index.md): internals, testing, transformers compatibility.
- [CLAUDE.md](CLAUDE.md) routes a task to the right page.
- [GAPS.md](GAPS.md) maps the nnterp 1.x API onto this one, for users of nnterp 1.x.

## I found a bug

Before opening an issue, reduce it to a minimal working example. If you can, write the
same code with nnsight's `TransformersModel` on the native module names: if that also fails,
the bug is in nnsight, so open it on the [nnsight tracker](https://github.com/ndif-team/nnsight/issues).
Also check that the model loads with `AutoModelForCausalLM` from transformers. Then open an
issue on the [nnterp tracker](https://github.com/ndif-team/nnterp/issues).

## Contributing

Contributions are welcome: a family that is not supported yet, a value you needed for your
research, a fix. Read [docs/developing/contributing.md](docs/developing/contributing.md) for
the style and workflow, and the list of open items.

## Citation

If you use `nnterp` in your research, you can cite it as:

```bibtex
@misc{dumas2025nnterp,
      title={nnterp: A Standardized Interface for Mechanistic Interpretability of Transformers},
      author={Cl{\'e}ment Dumas},
      year={2025},
      eprint={2511.14465},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.14465},
}
```
