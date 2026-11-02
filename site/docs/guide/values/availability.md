# Availability

Not every checkpoint has every value. A model loaded with `sdpa` never builds an attention pattern, and OPT has no `mlp` module. `model.support()` says which values this checkpoint has before any trace, from the config and the module tree alone. `None` means available, and anything else is the reason, a string. A family's page in the [encyclopedia](https://ndif-team.github.io/nnterp/encyclopedia/) shows the same report for each of its public checkpoints, with the reason beside every value the checkpoint lacks.

```python
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", dispatch=True, device="cpu")
support = model.support()
print(support["layer_output"])
print(support["self_attn.attention_output"])
print(support["self_attn.attention_probabilities"][0])
print(len(support["self_attn.attention_probabilities"]))
# None
# None
# read inside the eager attention forward, but this model runs 'sdpa'; load with attn_implementation='eager'
# 12
```

Loaded with `attn_implementation="eager"`, every entry on GPT-2 is `None`.

## Three forms

`model.support()` is the root's values plus every block value, by dotted name. A block value is `None` when every block has it, else `{layer: reason}` for the blocks that lack it, which is why the entry above is a dict with 12 keys. `model.support(layer=i)` is one block, flat, and `envoy.support()` is one envoy's own values without the prefix.

```python
print(model.support(layer=0)["self_attn.attention_scores"])
print(model.layers[0].mlp.support())
# read inside the eager attention forward, but this model runs 'sdpa'; load with attn_implementation='eager'
# {'mlp_activation': None, 'mlp_neurons': None, 'mlp_output': None}
```

The keys come from the tree. Every child of a block that carries standard values is listed under its standard name, so a value you add through `envoys=` appears as `self_attn.<name>` or `mlp.<name>`. A module that some blocks lack is reported on those blocks, and a module no block has (OPT's `mlp`) has no key at all, so guard with `support.get("mlp.mlp_output")` on a checkpoint that may lack it. A hybrid reads as a per-block dict. GPT-2 has every module, which is why this block uses Qwen3.5-0.8B, where each block carries either `linear_attn` or `self_attn`.

```python
hybrid = StandardizedTransformer("Qwen/Qwen3.5-0.8B", attn_implementation="eager")
support = hybrid.support()
print(sorted(support["self_attn.attention_output"]))
print(support["self_attn.attention_output"][0])
print(sorted(support["linear_attn.attention_output"]))
print([i for i, layer in enumerate(hybrid.layers) if getattr(layer, "self_attn", None) is not None])
# [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22]
# no self_attn module on this block
# [3, 7, 11, 15, 19, 23]
# [3, 7, 11, 15, 19, 23]
```

No `dispatch=True` here: `support()` reads the config and the tree, which a meta-device model has.

## The reasons

| reason, as it starts | values | when |
| --- | --- | --- |
| `read inside the eager attention forward, but this model runs 'sdpa'` | the attention interior | loaded with any `attn_implementation` but `"eager"` |
| `this checkpoint sets reorder_and_upcast_attn` | the GPT-2 interior | a GPT-2 config with that flag, which takes GPT-2's own attention path |
| `no self_attn module on this block`, `no linear_attn module on this block` | every value of that module | a hybrid's blocks: some other block has the module and this one does not |
| `a parallel block: the attention and the MLP read the same input, so there is no stream between them` | `layer_mid` | a parallel block (Cohere, GPT-NeoX with `use_parallel_residual`): the [contribution identity](residual-stream.md#between-the-sublayers) still holds, and only `layer_mid` has no tensor |
| `no act_fn module on this mlp`, `no down_proj module on this mlp` | `mlp_activation`, `mlp_neurons` | the family aliases no module to that name, such as an activation applied as a function |
| `routed experts; read per-expert via the Moe component` | `mlp_activation`, `mlp_neurons` | the block's MLP is a mixture of experts ([Mixture of experts](../architectures/mixture-of-experts.md)) |
| `the state after each token is materialized only by the token-by-token kernel` | `linear_attn.state`, `linear_attn.states` | a gated DeltaNet mixer before `nnterp.route_kernels(model.family, "torch")` ([Gated DeltaNet](../architectures/delta-net.md)) |
| `the default family cannot tell what this sublayer adds to the stream` | `attention_output`, `mlp_output` | a `model_type` with no family ([Loading](../loading.md#a-model_type-with-no-family)) |

The reasons are evaluated on the instance, so a change after the load changes the answer: `route_kernels` turns the `state` entries to `None`. `support()` predicts from the config; if a run's forward then takes a path the family does not expect, the read raises nnsight's `SourceNotAvailable` naming the missing operation ([Intermediate Operations](https://nnsight.net/features/11_source/)).

## `nnterp.Unavailable` at the read

Reading or writing an unavailable value raises `nnterp.Unavailable` with the same reason, at that line, before the model runs.

```python
import nnterp

try:
    with model.trace("The Eiffel Tower is in"):
        pattern = model.layers[0].self_attn.attention_probabilities.save()
except nnterp.Unavailable as error:
    print(error)
# model.transformer.h.0.attn.attention_probabilities is not available: read inside the eager attention forward, but this model runs 'sdpa'; load with attn_implementation='eager'
```

A module that does not exist on a block is an ordinary missing attribute: `model.layers[0].mlp` on OPT raises `AttributeError`, and `getattr(layer, "mlp", None)` is `None`, which is the form to pick blocks with outside the trace.

## `hasattr`

`hasattr` treats only `AttributeError` as absence. An unavailable value raises `Unavailable` through it, and an available value read outside a trace raises nnsight's `ValueError`, so `hasattr` never answers `False`. Check `support()` outside the trace instead.

```python
try:
    hasattr(model.layers[0].self_attn, "attention_probabilities")
except nnterp.Unavailable as error:
    print(type(error).__name__)
try:
    hasattr(model.layers[0], "layer_output")
except ValueError as error:
    print(error)
# Unavailable
# Cannot access `model.transformer.h.0.output` outside of interleaving
```

## Guarding code

Branch on `support()` outside the trace, and the body reads only what exists.

```python
model = StandardizedTransformer("openai-community/gpt2", dispatch=True, device="cpu", attn_implementation="eager")
support = model.support()
has_pattern = support["self_attn.attention_probabilities"] is None
has_mlp = support.get("mlp.mlp_output", "absent") is None

with model.trace("The Eiffel Tower is in"):
    if has_pattern:
        pattern = model.layers[0].self_attn.attention_probabilities.save()
    if has_mlp:
        mlp = model.layers[0].mlp.mlp_output.save()
print(pattern.shape, mlp.shape)
# torch.Size([1, 12, 7, 7]) torch.Size([1, 7, 768])
```

The repr lists a value whether or not this checkpoint has it. Only a value a family marks unavailable in its class body prints as `Unavailable: <reason>`, and a config-dependent reason shows in `support()` alone. Once a value is available, its shape is on [Layouts](layouts.md).
