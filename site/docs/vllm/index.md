# Loading and what differs

`StandardizedVLLM(repo_id, dispatch=True, ...)` is nnsight's [`VLLM`](https://nnsight.net/vllm/) with the step that `StandardizedTransformer` adds before the load: it reads the checkpoint's config, takes the family from `model_type` under `nnterp.families.vllm`, and hands that family's aliases and envoy classes to nnsight. The [names](../guide/names.md), the values and their [layouts](../guide/values/layouts.md) are those of `StandardizedTransformer` with a batch axis of 1. A block written against `model.layers[i].layer_output[:, -1]` runs on either engine.

```python
import torch
from nnterp import StandardizedVLLM

model = StandardizedVLLM("Qwen/Qwen2.5-0.5B", dispatch=True, dtype="float32", gpu_memory_utilization=0.3)
print(model.family.__name__, model.num_layers, model.hidden_size, model.num_heads, model.num_kv_heads)

block = model.layers[12]
with model.trace("The Eiffel Tower is in", temperature=0.0, max_tokens=1):
    x = block.layer_input.save()
    attn = block.self_attn.attention_output.save()
    mlp = block.mlp.mlp_output.save()
    out = block.layer_output.save()
    logits = model.logits.save()
print(x.shape, attn.shape, mlp.shape, out.shape)
print(logits.shape, repr(model.tokenizer.decode(logits[0, -1].argmax())))
print((x + attn + mlp - out).abs().max().item())
# nnterp.families.vllm.qwen2 24 896 14 2
# torch.Size([1, 7, 896]) torch.Size([1, 7, 896]) torch.Size([1, 7, 896]) torch.Size([1, 7, 896])
# torch.Size([1, 1, 151936]) ' Paris'
# 0.0
```

The load is in float32 so that the comparison at the end of this page holds at float32's resolution. Without `dtype` the weights load in the config's dtype, bfloat16 here. `gpu_memory_utilization` is vLLM's share of the card, which the reader sets for their own.

## Arguments

`family=`, `rename=` and `envoys=` are nnterp's own and merge as on [Loading](../guide/loading.md#arguments), `envoys=` over the family's `ENVOYS` and nnsight's envoys for vLLM's parallel layers. Everything else, such as `dispatch`, `dtype`, `gpu_memory_utilization`, `max_model_len`, `tensor_parallel_size`, `mode` and `taps`, goes to `VLLM` as documented at [Loading models](https://nnsight.net/vllm/loading/), and the full signature is under the Documentation tab ([standardized_vllm](../documentation/standardized_vllm.md)). There is no `attn_implementation`: vLLM's attention kernels are its own, and the values inside attention are served another way ([below](#the-attention-interior)). In a `.py` file the engine must be built under a `__main__` guard, because vLLM spawns its worker and the spawn re-imports the main module ([nnsight](https://nnsight.net/vllm/loading/#main-guard)). The block runs in that worker, where nnterp has to be importable.

## `layer_input` and `layer_output`

On most families a block is called `forward(positions, hidden_states, residual)` and returns `(hidden_states, residual)`, because vLLM fuses each residual add into the norm that follows, and the residual stream is the pair's sum ([Locations](https://nnsight.net/vllm/locations/#tuples-and-the-residual-stream)). Natively, `layers[i].input` is the positions and `layers[i].output` the pair. `layer_input` and `layer_output` are the stream on every family and on both engines.

```python
with model.trace("The Eiffel Tower is in", temperature=0.0, max_tokens=1):
    positions = block.input.clone().save()
    hidden = block.output[0].clone().save()
    residual = block.output[1].clone().save()
    stream = block.layer_output.save()
print(positions)
print(hidden.shape, residual.shape, torch.equal(hidden + residual, stream[0]))
# tensor([0, 1, 2, 3, 4, 5, 6], device='cuda:0')
# torch.Size([7, 896]) torch.Size([7, 896]) True
```

## One sequence, batch axis 1

A request is one sequence. vLLM serves the request's own rows, `[tokens, ...]`, and nnterp puts the batch axis back, always 1. Under `tracer.iter` step 0 is the prefill, every prompt token, and each later step one token. `logits` is `[1, 1, vocab_size]`, because the engine computes the last position only. `logits[:, -1]` and `next_token_probs` read the same on both engines, and nnsight's `model.samples` is the token drawn from them ([Generation](https://nnsight.net/vllm/generation/)). Several prompts are several invokes, one prompt each, and a name saved in each comes back as a list.

```python
import nnsight

with model.trace("The Eiffel Tower is in", temperature=0.0, max_tokens=3, ignore_eos=True) as tracer:
    shapes = nnsight.save([])
    for step in tracer.iter[:3]:
        shapes.append((tuple(block.layer_output.shape), tuple(model.logits.shape)))
    result = tracer.result.save()
print(shapes)
print(repr(result.outputs[0].text))
# [((1, 7, 896), (1, 1, 151936)), ((1, 1, 896), (1, 1, 151936)), ((1, 1, 896), (1, 1, 151936))]
# ' Paris, France'
```

A trace is a generation request, and the sampling settings go on `trace` or `invoke`. With none, vLLM's defaults apply, 16 tokens at temperature 1.0. The blocks on this page pass `temperature=0.0, max_tokens=1` for one deterministic step. A block without `tracer.iter` runs on the prefill whatever `max_tokens` is.

```python
with model.trace("The Eiffel Tower is in") as tracer:
    result = tracer.result.save()
print(len(result.outputs[0].token_ids))
# 16
```

## Private copies

A tensor vLLM serves is the model's live buffer, which a later kernel may rewrite ([Clone what you keep](https://nnsight.net/vllm/locations/#clone-what-you-keep)). Every nnterp value on this engine is a private copy, handed back to the model when the block moves on: a saved value stays what it was read as without a `.clone()`, and an in-place edit and an assignment both reach the model. An assigned value must keep the shape it was served with, because the rows are spliced into a step the engine runs for every request in flight. nnterp refuses another height in the block that wrote it, the request fails, and the engine carries on. An error raised in the worker arrives as a `RuntimeError` naming the original type.

```python
with model.trace("The Eiffel Tower is in", temperature=0.0, max_tokens=1):
    clean = model.logits.save()

with model.trace("The Eiffel Tower is in", temperature=0.0, max_tokens=1):
    block.layer_output[:, -1] *= 0.5
    model.layers[13].mlp.mlp_output = model.layers[13].mlp.mlp_output * 0
    edited = model.logits.save()
print((edited - clean).abs().max().item())
# 10.073400497436523

try:
    with model.trace("The Eiffel Tower is in", temperature=0.0, max_tokens=1):
        block.layer_output = block.layer_output[:, :-1]
except RuntimeError as error:
    print(str(error).splitlines()[0])
# ValueError: model.model.layers.12.layer_output is (1, 7, 896) on this request and cannot be replaced by a value of shape (1, 6, 896); write into the rows you mean instead (``value[:, positions] = ...``)
```

## The attention interior

`attention_queries`, `attention_keys`, `attention_values` and `attention_head_outputs` are what enters and leaves vLLM's attention layer, in the layouts of [Attention](../guide/values/attention.md), and writable like any value. The scores and the pattern are computed inside that layer's kernel, and nnterp recomputes them from the queries and keys with the layer's own scale, softcap and sliding window. They are read-only, and there on the prefill only, since a decode step holds one token's keys ([What is unavailable](unavailable.md#the-pattern-on-a-decode-step)).

```python
attn = block.self_attn
with model.trace("The Eiffel Tower is in", temperature=0.0, max_tokens=1):
    q = attn.attention_queries.save()
    k = attn.attention_keys.save()
    v = attn.attention_values.save()
    scores = attn.attention_scores.save()
    pattern = attn.attention_probabilities.save()
    heads = attn.attention_head_outputs.save()
print(q.shape, k.shape, v.shape)
print(scores.shape, pattern.shape, heads.shape)
print(torch.allclose(scores.float().softmax(-1), pattern), pattern[0, 0].sum(-1))
# torch.Size([1, 14, 7, 64]) torch.Size([1, 2, 7, 64]) torch.Size([1, 2, 7, 64])
# torch.Size([1, 14, 7, 7]) torch.Size([1, 14, 7, 7]) torch.Size([1, 7, 14, 64])
# True tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
#        device='cuda:0')
```

## The same values on both engines

The suite under `tests/vllm_families/` holds every value the engine serves against `StandardizedTransformer`'s on the same checkpoint and prompt, both in float32. On Qwen2.5-0.5B, block 12's stream agrees to 1e-5 of its largest magnitude, the pattern to 1e-3, the head outputs to 4e-3 and the logits to 2e-3. The two engines' attention kernels sum a sharp softmax differently, and that difference is what the later values carry. Compare at 5e-3 of a value's scale, and expect more from a bfloat16 load.

```python
from nnterp import StandardizedTransformer

hf = StandardizedTransformer("Qwen/Qwen2.5-0.5B", device="cuda", dtype=torch.float32, attn_implementation="eager")
with torch.no_grad(), hf.trace("The Eiffel Tower is in"):
    hf_pattern = hf.layers[12].self_attn.attention_probabilities.save()
    hf_heads = hf.layers[12].self_attn.attention_head_outputs.save()
    hf_out = hf.layers[12].layer_output.save()
    hf_logits = hf.logits[:, -1:].save()

def gap(vllm_value, hf_value):
    """The largest difference, relative to the transformers value's largest magnitude."""
    return ((vllm_value.cpu() - hf_value.cpu()).abs().max() / hf_value.cpu().abs().max()).item()

print(out.shape == hf_out.shape, logits.shape == hf_logits.shape)
print(f"{gap(out, hf_out):.1e} {gap(pattern, hf_pattern):.1e} {gap(heads, hf_heads):.1e} {gap(logits, hf_logits):.1e}")
print(logits.argmax().item() == hf_logits.argmax().item())
# True True
# 1.2e-05 9.5e-04 4.1e-03 1.9e-03
# True
```

The values the engine cannot serve are on [What is unavailable](unavailable.md).
