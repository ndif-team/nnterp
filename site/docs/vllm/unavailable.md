# What is unavailable

`model.support()` on a vLLM engine reports the values the engine cannot serve, each with its reason as a string, and `None` for the rest, as on transformers ([Availability](../guide/values/availability.md)). On a checkpoint with a gated MLP, such as Qwen2.5-0.5B, there are three.

```python
from nnterp import StandardizedVLLM

model = StandardizedVLLM("Qwen/Qwen2.5-0.5B", dispatch=True, dtype="float32", gpu_memory_utilization=0.3)
for name, reason in model.support().items():
    if reason:
        print(name)
        print("   ", reason if isinstance(reason, str) else reason[0])
# lm_head_output
#     vLLM never calls lm_head: its logits processor projects with lm_head's weight and applies any softcap and scale in the same call, on the last position only; read logits
# attention_mask
#     a vLLM request is one unpadded sequence; there is no mask
# mlp.mlp_activation
#     a gated MLP on vLLM: act(gate) is never a module's output (gate_up_proj feeds act_fn, an act-and-multiply kernel, whose output is already act(gate) * up), so it is not served; mlp_neurons is the product
```

Reading or writing one of them raises `nnterp.Unavailable` with the same reason at that line, as on transformers. Because the block runs in the engine's worker, the error arrives as a `RuntimeError` naming `Unavailable`.

## A gated MLP's `mlp_activation`

vLLM fuses the gate and up projections into one `gate_up_proj`, and its `act_fn` (`SiluAndMul`, `GeluAndMul`) computes `act(gate) * up` in one kernel, and `act(gate)` alone is never a tensor. `mlp_neurons`, the down projection's input, is served and is that product.

```python
mlp = model.layers[12].mlp
with model.trace("The Eiffel Tower is in", temperature=0.0, max_tokens=1):
    neurons = mlp.mlp_neurons.save()
print(neurons.shape, type(mlp.act_fn._module).__name__)
# torch.Size([1, 7, 4864]) SiluAndMul

try:
    with model.trace("The Eiffel Tower is in", temperature=0.0, max_tokens=1):
        activation = mlp.mlp_activation.save()
except RuntimeError as error:
    print(str(error).splitlines()[0][:80])
# Unavailable: model.model.layers.12.mlp.mlp_activation is not available: a gated
```

A plain MLP, such as GPT-2's, serves both as the same tensor. Both are read at the down projection: vLLM builds one activation module by name and shares it between every block, and that module's output cannot tell the blocks apart.

```python
gpt2 = StandardizedVLLM("openai-community/gpt2", dispatch=True, dtype="float32", gpu_memory_utilization=0.15)
print(sorted(name for name, reason in gpt2.support().items() if reason))
# ['attention_mask', 'lm_head_output']
```

## `lm_head_output`

vLLM never calls `lm_head`. Its logits processor projects with the head's weight and applies any softcap and scale in the same call, for the last position only. Read `logits`, or run `project_on_vocab` on a stream: on the last block's stream it is `logits`, as on transformers. The module is still on the tree. `project_on_vocab` and `get_topk_closest_tokens` call the engine's modules and work inside a trace only.

```python
with model.trace("The Eiffel Tower is in", temperature=0.0, max_tokens=1):
    lens = model.project_on_vocab(model.layers[-1].layer_output[:, -1:]).save()
    logits = model.logits.save()
print(lens.shape, (lens - logits).abs().max().item())
# torch.Size([1, 1, 151936]) 4.76837158203125e-06
```

## The pattern on a decode step

`attention_scores` and `attention_probabilities` are recomputed from the step's queries and keys ([The attention interior](index.md#the-attention-interior)). On the prefill the step holds every key. On a decode step the new token's key is the only one served, and the earlier ones are in vLLM's cache, which only its kernel reads, and the read raises before anything runs. A trace without `tracer.iter` runs its block on the prefill whatever `max_tokens` is, and `support()`, which runs outside any step, lists both as available. Both are read-only, since the kernel never takes a pattern: to change what a head attends to, edit its queries or keys, and to change what it wrote, edit `attention_head_outputs`.

```python
attn = model.layers[12].self_attn
with model.trace("The Eiffel Tower is in", temperature=0.0, max_tokens=3, ignore_eos=True):
    pattern = attn.attention_probabilities.save()
print(pattern.shape)
# torch.Size([1, 14, 7, 7])

try:
    with model.trace("The Eiffel Tower is in", temperature=0.0, max_tokens=2, ignore_eos=True) as tracer:
        for step in tracer.iter[1]:
            pattern = attn.attention_probabilities.save()
except RuntimeError as error:
    print(str(error).splitlines()[0])
# Unavailable: model.model.layers.12.self_attn.attention_probabilities is not available: recomputed from this step's queries and keys, and on a decode step the earlier tokens' keys are in vLLM's cache, which only its kernel reads; read it on the prefill

try:
    with model.trace("The Eiffel Tower is in", temperature=0.0, max_tokens=1):
        attn.attention_probabilities = attn.attention_probabilities * 0
except RuntimeError as error:
    print(str(error).splitlines()[0])
# AttributeError: attention_probabilities is derived and read-only
```

## `attention_mask` and the read-only inputs

A request is one unpadded sequence with no mask, and `input_size` is `[1, tokens]`. `input_ids` and `input_size` are read-only on this engine, and `next_token_probs` is derived from `logits` on both engines. Assign `logits` to change what the sampler sees.

```python
with model.trace("The Eiffel Tower is in", temperature=0.0, max_tokens=1):
    ids = model.input_ids.save()
    size = model.input_size.save()
print(ids, size)
# tensor([[  785,   468,  3092,   301, 21938,   374,   304]], device='cuda:0',
#        dtype=torch.int32) torch.Size([1, 7])

try:
    with model.trace("The Eiffel Tower is in", temperature=0.0, max_tokens=1):
        model.next_token_probs = model.next_token_probs * 0
except RuntimeError as error:
    print(str(error).splitlines()[0])
# AttributeError: next_token_probs is derived from the logits and cannot be assigned; assign model.logits instead
```

## On some families

A mixture of experts serves `mlp_output` and nothing inside it: the experts run in vLLM's fused MoE kernel, so `mlp_activation` and `mlp_neurons` are unavailable, and the router has no values on this engine. DeepSeek-V2, DeepSeek-V3, Kimi-K2 and GLM-4.7-Flash run latent attention in a kernel of their own, so the six values of the attention interior are unavailable there while `attention_output` is served. The two reasons are constants, which `support()` prints for the affected values.

```python
from nnterp.components.vllm.mlp import ROUTED
from nnterp.families.vllm.deepseek_v2 import LATENT

print(ROUTED)
print(LATENT)
# routed experts, inside vLLM's fused MoE kernel: there is no single activation or down projection
# vLLM runs DeepSeek's latent attention in its own kernel, on compressed keys and values; not mapped on vLLM
```

## Methods and nnsight's limits

`model.order` and `model.rank` are measured on `StandardizedTransformer` only. Gradients, `.source` inside a kernel and `scan` are nnsight's limits on this engine ([Capabilities and limits](https://nnsight.net/vllm/capabilities/#not-on-vllm)). `skip_layers` has to answer for every row of a step, and a step holds whatever requests the engine batched together: it is safe on a request that runs alone. To ablate a block on a shared engine, write its contributions to zero instead.

```python
try:
    model.order(12)
except NotImplementedError as error:
    print(error)
# model.order is not supported on the vllm engine; it is measured on a StandardizedTransformer
```

Which families run on this engine, and what each changes, is on [Families on vLLM](families.md).
