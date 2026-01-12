<p align="center">
  <img src="./nnterp_logo.png" alt="nnterp" width="300">
</p>

<h3 align="center">
A unified interface for mechanistic interpretability of transformers
</h3>

<p align="center">
<i>Created by Clément Dumas</i>
</p>

<p align="center">
| <a href="https://ndif-team.github.io/nnterp/"><b>Documentation</b></a> | <a href="https://github.com/ndif-team/nnterp"><b>GitHub</b></a> | <a href="https://discord.gg/6uFJmCSwW7"><b>Discord</b></a> | <a href="https://discuss.ndif.us/"><b>Forum</b></a> | <a href="https://x.com/ndif_team"><b>Twitter</b></a> | <a href="https://arxiv.org/abs/2511.14465"><b>Paper</b></a> |
</p>

<p align="center">
<a href="https://colab.research.google.com/github/Butanium/nnterp/blob/main/demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></img></a>
</p>

---

## About

**nnterp** is a unified interface for all transformer models that puts best NNsight practices for LLMs in everyone's hands.

Built on top of [NNsight](https://github.com/ndif-team/nnsight), nnterp provides a standardized interface for mechanistic interpretability research across all transformer architectures. Unlike `transformer_lens` which reimplements transformers, `nnterp` preserves the original HuggingFace implementations while solving the naming convention chaos through intelligent renaming.

## Why nnterp?

**The Problem**: Every transformer model uses different naming conventions - GPT-2 uses `transformer.h`, LLaMA uses `model.layers`, OPT uses something else entirely. This makes mechanistic interpretability research painful as you can't just change the name of the model and expect the rest of your code to work.

**The Solution**: `nnterp` standardizes all models to use something close to the LLaMA naming convention:
```
StandardizedTransformer
├── layers
│   ├── self_attn
│   └── mlp
├── ln_final
└── lm_head
```
and include built-in properties like `model.logits` and `model.next_token_probs`.

Unlike other libraries that reimplement transformers, `nnterp` uses `NNsight`'s renaming feature to work with the original HuggingFace implementations, ensuring perfect compatibility and preventing subtle bugs. `nnterp` also includes automatic testing to ensure models are correctly standardized. When you load a model, it runs fast validation checks. See the [documentation](https://ndif-team.github.io/nnterp/model-validation.html) for details.

## Installation
- `pip install nnterp` - Basic installation
- `pip install nnterp[display]` - Includes visualization dependencies
<!-- - `pip install nnterp[vllm]` - Includes vLLM support for efficient inference (NOT supported yet) -->


## Examples

Here is a simple example where we load a model and access its standardized internals:

```python
from nnterp import StandardizedTransformer

model = StandardizedTransformer("gpt2")  # or "meta-llama/Llama-2-7b-hf", etc.

with model.trace("The Eiffel Tower is in the city of"):
    # Unified interface across all models (must follow forward pass order!)
    attention_output = model.attentions_output[3]
    mlp_output = model.mlps_output[3]
    layer_5_output = model.layers_output[5]

    # Built-in utilities
    logits = model.logits.save()
```

---

### Standardized Naming

All models use the same naming convention:

```python
with model.trace("Hello world"):
    # Attention and MLP components (access in forward pass order!)
    attn_out = model.attentions_output[3]
    mlp_out = model.mlps_output[3]
    layer_3_output = model.layers_output[3]

    # Layer I/O - works for GPT-2, LLaMA, Gemma, etc.
    layer_5_output = model.layers_output[5]

    # Direct interventions - add residual from layer 3 to layer 10
    model.layers_output[10] = model.layers_output[10] + layer_3_output
```

---

### Built-in Interventions

Common mechanistic interpretability interventions with best practices built-in:

```python
from nnterp.interventions import logit_lens, patchscope_lens, steer

# Logit lens: decode hidden states at each layer
layer_probs = logit_lens(model, ["The capital of France is"])
# Shape: (batch, layers, vocab_size)

# Patchscope: patch hidden states across prompts
from nnterp.interventions import TargetPrompt
target = TargetPrompt("The capital of France is", index_to_patch=-1)
patchscope_probs = patchscope_lens(
    model,
    source_prompts=["The capital of England is"],
    target_patch_prompts=target,
    layer_to_patch=10
)

# Activation steering
import torch
with model.trace("Hello, how are you?"):
    steering_vector = torch.randn(model.hidden_size)
    model.steer(layers=[5, 10], steering_vector=steering_vector, factor=1.5)
```

---

### Prompt and Target Tracking

Track probabilities for specific tokens across interventions:

```python
from nnterp.prompt_utils import Prompt, run_prompts

prompts = [
    Prompt.from_strings(
        "The capital of France is",
        {"target": "Paris", "other": ["London", "Madrid"]},
        model.tokenizer
    )
]

# Get probabilities for all target categories
results = run_prompts(model, prompts)
# Returns: {"target": tensor([0.85]), "other": tensor([0.12])}

# Combine with interventions
results = run_prompts(model, prompts, get_probs_func=logit_lens)
# Returns probabilities across all layers
```
---

More examples and detailed documentation can be found at [butanium.github.io/nnterp](https://ndif-team.github.io/nnterp/)


## I found a bug!

Before opening an issue, make sure that you have a MWE (minimal working example) that reproduces the issue, and if possible, the equivalent code using `NNsight.LanguageModel`. If the NNsight MWE also fails, please open an issue on the [NNsight repository](https://github.com/ndif-team/nnsight/issues/). Also make sure that you can load the model with `AutoModelForCausalLM` from `transformers`.


## Contributing
Contribution are welcome! If a functionality is missing, and you implemented it for your reasearch, please open a PR, so that people in the community can benefit from it. That include adding support for new models with custom renamings!

## Next steps
Here are some nice features that could be cool to have, and for which I'd be happy to accept PRs (ordered by most to least useful imo):
- [ ] Add helpers for getting gradients
- [ ] Add support for `vllm` when `NNsight` supports it
- [ ] Add helpers for `NNsight`'s cache as it returns raw tuple outputs instead of nice vectors.
- [ ] Add access to k/q/v

## Development
- Install the development environment with `make dev` or `uv sync --all-extras`. Add `uv pip install flash-attn --no-build-isolation` to support models like `Phi` that require `flash-attn`.
- Install pre-commit hooks with `pre-commit install` to automatically update `docs/llms.txt` when modifying RST files and format the code with `black`.
You might encounter the error `with block not found at line xyz` when running the tests. In this case run `make clean` to remove the python cache and try again (NOTE: this should be fixed in latest `NNsight` versions).
- Create a git tag with the version number `git tag vx.y.z; git push origin vx.y.z`
- Build with `python -m build`
- Publish with e.g. `twine upload dist/*x.y.z*`
- test with `pytest --cache-clear`. **cache-clear is mandatory for now otherwise `NNsight`'s source can break.** It might not be sufficient, in which case you can do `make clean` to remove Python cache.


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