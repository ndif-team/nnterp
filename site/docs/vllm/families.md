# Families on vLLM

A vLLM family is one module under `nnterp/families/vllm/`, named after the `model_type` and written against vLLM's own implementation of that architecture. `nnterp.families.known("vllm")` lists the 60 shipped ones, and `lookup(model_type, engine="vllm")` resolves a type as the transformers lookup does ([Registering](../developing/registering.md#the-lookup-order)) with one difference. The best-effort `default` family is written against transformers' modules, and a `model_type` with no vLLM family raises `UnsupportedFamily` instead of falling back to it.

```python
from nnterp import UnsupportedFamily, families

on_vllm = families.known("vllm")
print(len(on_vllm), len(families.known()))
print(families.lookup("qwen2", engine="vllm").__name__, families.lookup("qwen2").__name__)
try:
    families.lookup("diffllama", engine="vllm")
except UnsupportedFamily as error:
    print(str(error)[:90])
# 60 98
# nnterp.families.vllm.qwen2 nnterp.families.qwen2
# no standardization for model_type 'diffllama' on vllm; known: ['afmoe', 'apertus', 'arcee'
```

## The module

A vLLM family declares what a transformers family declares, `RENAME`, `Layer`, `Attention`, `Mlp` and `ENVOYS` ([Adding a family](../developing/adding-a-family.md)), with `ENVOYS` keyed on vLLM's module classes and the three envoys subclassing the bases in `nnterp.components.vllm`, which read each value where vLLM keeps it and serve it as a private `[1, tokens, ...]` copy. Qwen2's module is the whole of `nnterp/families/vllm/qwen2.py`:

```python
"""Qwen2 and Qwen2.5 on vLLM (``vllm.model_executor.models.qwen2``).

Llama's names and Llama's fused block, in classes of its own.
"""

from vllm.model_executor.models.qwen2 import Qwen2Attention, Qwen2DecoderLayer, Qwen2MLP

from nnterp.components.vllm import Attention, FusedLayer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(FusedLayer):
    """Qwen2's decoder block: ``forward(positions, hidden_states, residual) -> (hidden_states, residual)``."""


class Attention(Attention):
    """Qwen2's attention; its output is what the block adds, so the base holds."""


class Mlp(Mlp):
    """Qwen2's MLP; its output is what the block adds, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Qwen2DecoderLayer: Layer, Qwen2Attention: Attention, Qwen2MLP: Mlp}
```

What the family has to say is which convention its block follows, because the return type does not tell. `FusedLayer` is a block that takes and returns `(hidden_states, residual)` whose sum is the stream, with `HIDDEN` and `RESIDUAL` saying where the halves sit in the call. `Layer` is a block called with the stream, alone or after the positions (`STREAM`), that returns it alone or with something beside it (`returns_tuple`). EXAONE 4's block is called `forward(positions, hidden_states, residual)` and returns a pair, and is a `Layer` all the same, because the pair's first element is the whole stream. Where what the block adds is not a module's output, the family redefines the value with `Flat`, the descriptor for a `[tokens, ...]` tensor: the post-norm's output on Gemma-2 and OLMo-2, the module's output times `residual_multiplier` on Granite. The signatures are under the Documentation tab ([vllm components](../documentation/components/vllm/index.md)).

```python
from nnterp.components.vllm import FusedLayer, Layer
from nnterp.components.vllm.transformers_backend import Layer as BackendLayer
from nnterp.families.vllm import exaone4, gpt2, llama, olmo2, phi

print(issubclass(llama.Layer, FusedLayer), llama.Layer.HIDDEN, llama.Layer.RESIDUAL)
print(issubclass(gpt2.Layer, Layer), gpt2.Layer.STREAM, gpt2.Layer.returns_tuple)
print(issubclass(phi.Layer, Layer), phi.Layer.STREAM, phi.Layer.returns_tuple)
print(issubclass(exaone4.Layer, Layer), exaone4.Layer.STREAM, exaone4.Layer.returns_tuple)
print(issubclass(olmo2.Layer, BackendLayer), list(olmo2.ENVOYS)[0].__module__)
# True (1, 'hidden_states') (2, 'residual')
# True 0 False
# True 1 False
# True 1 True
# True transformers.models.olmo2.modeling_olmo2
```

## The transformers backend

vLLM has no implementation of its own for OLMo, OLMo 2, SmolLM3, StarCoder2 and GPT-BigCode and runs transformers' modules inside the engine. Those five families key `ENVOYS` on transformers' classes and build on `nnterp.components.vllm.transformers_backend`, where the activations already carry the batch axis and the engine's attention layer, which the backend keeps outside the module tree, is mounted as `self_attn.attn` inside a trace, and the queries, keys, values and head outputs are then served as on every other vLLM family. `.source` does not work on these modules, because the backend rewrites their forwards. Passing the module as `family=` is how a family is tried before it ships, as on transformers ([Checking the module](../developing/adding-a-family.md#checking-the-module-before-it-ships)):

```python
import torch
from nnterp import StandardizedVLLM
from nnterp.families.vllm import olmo2

model = StandardizedVLLM("allenai/OLMo-2-0425-1B", family=olmo2, dispatch=True, dtype="float32", gpu_memory_utilization=0.3)
print(model.family is olmo2, type(model._module).__name__, type(model.layers[0]._module).__name__)
print(sorted(name for name, reason in model.support().items() if reason))

block = model.layers[8]
with model.trace("The Eiffel Tower is in", temperature=0.0, max_tokens=1):
    raw = block.input.clone().save()
    x = block.layer_input.save()
    q = block.self_attn.attention_queries.save()
    pattern = block.self_attn.attention_probabilities.save()
    attn = block.self_attn.attention_output.save()
    mlp = block.mlp.mlp_output.save()
    out = block.layer_output.save()
    logits = model.logits.save()
print(raw.shape, torch.equal(raw, x), q.shape, pattern.shape)
print((x + attn + mlp - out).abs().max().item(), repr(model.tokenizer.decode(logits[0, -1].argmax())))
# True TransformersForCausalLM Olmo2DecoderLayer
# ['attention_mask', 'attention_norm_output', 'lm_head_output', 'mlp.mlp_activation', 'mlp_norm_output']
# torch.Size([1, 7, 2048]) True torch.Size([1, 16, 7, 128]) torch.Size([1, 16, 7, 7])
# 0.0 ' Paris'
```

On OLMo 2's post-norm block, `attention_norm_output` and `mlp_norm_output` are unavailable with the post-norm reasons, as on transformers.

## Adding one

Write `nnterp/families/vllm/<model_type>.py` as above, reading vLLM's forward of the block before choosing the base. Then add `tests/vllm_families/test_vllm_<model_type>.py`, a `VLLMFamilySuite` subclass naming a checkpoint that both engines load. The suite runs the checkpoint through `StandardizedTransformer` in float32, builds the engine in float32, and holds every value it serves against the transformers value, the contribution identity and the edits included. Each class builds one engine, and each file runs in its own process. OLMo 2's test file is the whole of it:

```python
"""OLMo 2 on vLLM's transformers backend, against the transformers engine: transformers' modules, post-norm."""

from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import olmo2


class TestVLLMOlmo2(VLLMFamilySuite):
    REPO = "allenai/OLMo-2-0425-1B"
    FAMILY = olmo2
    UNAVAILABLE = VLLMFamilySuite.UNAVAILABLE | {"attention_norm_output", "mlp_norm_output"}
    NATIVE = LLAMA_ROWS
    MEMORY = 0.3
```

```
tests/vllm_families/run.sh olmo2
```

The checkpoint has to be one vLLM runs: its attention kernels need a head width of at least 16, and 32 for the float32 one, and most of its MLPs take SiLU only, which rules out most of the tiny random checkpoints the transformers suite pins. Name a small real checkpoint where one is published, and otherwise a tiny one with heads at least 32 wide, or a copy of one resized for vLLM that the test writes once (`test_vllm_hyperclovax.py`). A family kept outside the package registers with `nnterp.families.register(module, engine="vllm")`, and a family registered without `engine` is a transformers family only.

## Where the engines differ

vLLM's implementation of a family is not always transformers' model. Where the two disagree, nnterp serves what vLLM computes, and the family's test file aligns the two engines for the comparison and says how. On vLLM 0.27.1 and transformers 5.15:

| family | difference |
| --- | --- |
| GLM (`glm`) | vLLM rotates half of each head whatever `partial_rotary_factor` says; GLM-Edge's config says all of it |
| Ministral 3 | vLLM's YaRN ignores `mscale` and `mscale_all_dim` and scales every query and key by 1.277 at factor 16; an HF-format checkpoint needs `hf_overrides={"rope_parameters": {..., "apply_yarn_scaling": False}}` |
| GLM-4.7-Flash (`glm4_moe_lite`) | the latent norms use the config's `rms_norm_eps` where transformers fixes 1e-6 |
| PhiMoE | the logits leave out `lm_head.bias`, which vLLM loads and never adds |
| StableLM | vLLM builds the sequential block only, ignoring `use_parallel_residual` and `qk_layernorm` |
| FlexOlmo | the experts run SiLU whatever `hidden_act` says |
| Llama 4 | vLLM permutes each head's query and key channels at load, which the values undo; the attention temperature tuning follows the generation config, not the model config |
| DBRX | transformers reads the rotary base from `rope_parameters` only, where vLLM reads the checkpoint's `attn_config.rope_theta`; vLLM 0.27.1 loads no DBRX checkpoint |
| the kernels | given the same queries, keys and values, the two engines' float32 attention kernels differ by about 1e-3 of the head outputs' scale, and by 2.4e-2 on a softmax as sharp as Qwen2.5-0.5B's first block |

Each family's page in the [Documentation](../documentation/families/vllm/index.md) tab carries its own notes.
