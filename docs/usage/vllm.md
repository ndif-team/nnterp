---
title: The vLLM engine
one_liner: "`StandardizedVLLM` is nnsight's `VLLM` with nnterp's names and values: the same layouts as `StandardizedTransformer` (batch axis 1), private copies, vLLM's own defaults, and a family per vLLM implementation under `nnterp/families/vllm/`."
tags: [usage, vllm, engine, StandardizedVLLM, layer_input, families]
related: [docs/usage/loading.md, docs/usage/residual-stream.md, docs/usage/root-values.md, docs/usage/availability.md, docs/usage/generation.md, docs/extending/adding-a-family.md]
sources: [nnterp/standardized_vllm.py, nnterp/components/vllm/__init__.py, nnterp/components/vllm/flat.py, nnterp/components/vllm/layer.py, nnterp/components/vllm/attention.py, nnterp/families/vllm/__init__.py, nnterp/families/vllm/llama.py, nnterp/families/vllm/gemma2.py, nnterp/families/vllm/gpt2.py, tests/vllm_families/vllm_suite.py]
---

# The vLLM engine

## What this is for

`StandardizedVLLM` runs a checkpoint on nnsight's `VLLM` engine (continuous batching, tensor
parallelism, serving; nnsight `docs/models/vllm.md`) under nnterp's vocabulary. The names and
the values are `StandardizedTransformer`'s, with the same layouts, so a block written against
`model.layers[i].layer_output[:, -1]` runs on either engine.

Everything else is nnsight's `VLLM`, with vLLM's own defaults: a trace is a generation request
(`max_tokens=16`, `temperature=1.0` unless you say), sampling settings go on `trace` / `invoke`,
and each invoke is one prompt. Read nnsight's vLLM guide for the engine; this page is what
nnterp adds and what differs from the transformers engine.

Families on this engine: `llama`, `mistral`, `phi3`, `qwen2`, `qwen3`, `gemma`, `gemma2`,
`gemma3_text`, `exaone4`, `cohere`, `cohere2`, `mixtral`, `qwen2_moe`, `qwen3_moe`, `olmoe`,
`deepseek_v2`, `deepseek_v3`, `gpt2`, `gptj`, `gpt_neox`, `bloom`, `mpt`, `falcon`, `phi`,
`olmo3` ([family notes](#family-notes)). Another `model_type` raises `UnsupportedFamily`.

The snippets on this page ran on vLLM 0.27.1 with `HuggingFaceTB/SmolLM2-135M-Instruct`.

## Canonical pattern

```python
import nnsight
import torch
from nnterp import StandardizedVLLM

model = StandardizedVLLM("HuggingFaceTB/SmolLM2-135M-Instruct", dispatch=True,
                         gpu_memory_utilization=0.2, max_model_len=1024)

layer = model.layers[12]
with model.trace("The Eiffel Tower is in the city of", temperature=0.0, max_tokens=1):
    stream_in = layer.layer_input.save()                  # [1, tokens, hidden]
    attention = layer.self_attn.attention_output.save()   # [1, tokens, hidden]
    mlp = layer.mlp.mlp_output.save()                     # [1, tokens, hidden]
    stream = layer.layer_output.save()                    # [1, tokens, hidden]
    logits = model.logits.save()                          # [1, 1, vocab]: the last position
    probs = model.next_token_probs.save()                 # [1, vocab]

print(tuple(stream.shape), tuple(logits.shape), model.tokenizer.decode(logits[0, -1].argmax()))
# (1, 9, 576) (1, 1, 49152)  Paris
torch.testing.assert_close(stream_in + attention + mlp, stream)
```

A script needs an `if __name__ == "__main__":` guard around the model's construction: vLLM
starts its engine in a spawned process, which re-imports the main module.

## What is the same

- **The vocabulary**: `embed_tokens`, `layers[i].self_attn`, `layers[i].mlp`, `norm`, `lm_head`,
  beside vLLM's native names.
- **The boundary values and the identity**: `layer_input + attention_output + mlp_output ==
  layer_output` on every block; the suite holds each against the transformers engine's value
  for the same prompt.
- **The layouts**: `[batch, seq, hidden]` with a batch of 1, `logits[:, -1]` the last position.
- **The methods**: `steer`, `skip_layers`, `project_on_vocab`, `get_topk_closest_tokens`,
  `support()`, and the sizes (`num_layers`, `hidden_size`, `num_heads`, ...).
- **Reads, in-place edits and assignment** all reach the model.

```python
with model.trace("The Eiffel Tower is in the city of", temperature=0.0, max_tokens=1):
    lens = nnsight.save(model.get_topk_closest_tokens(model.layers[24].layer_output[:, -1], k=3))
    model.layers[25].layer_output[:, -1] *= 0.5            # in place
    model.layers[26].mlp.mlp_output = model.layers[26].mlp.mlp_output * 0   # assignment
    edited = model.logits.save()

print(list(lens[0]))
assert not torch.equal(edited, logits)
```

## What differs

### One sequence per request

vLLM serves a request's own rows, `[tokens, hidden]`; nnterp restores the batch axis, which is
always 1. Several prompts are several invokes, and a name saved in each comes back as a list:

```python
with model.trace(temperature=0.0, max_tokens=1) as tracer:
    for prompt in ["The capital of France is", "Two plus two is"]:
        with tracer.invoke(prompt):
            last = layer.layer_output[:, -1].save()        # [1, hidden], once per invoke

print(len(last), tuple(last[0].shape))
# 2 (1, 576)
```

Under `tracer.iter`, step 0 is the prefill (every prompt token) and each later step one token:

```python
with model.trace("The capital of France is", temperature=0.0, max_tokens=3, ignore_eos=True) as tracer:
    shapes = nnsight.save([])
    for step in tracer.iter[:3]:
        shapes.append(tuple(layer.layer_output.shape))

print(shapes)
# [(1, 5, 576), (1, 1, 576), (1, 1, 576)]
```

### Values are private copies

A tensor vLLM serves is the model's live buffer, which the next fused kernel rewrites; raw
`.output` saved without a clone comes back holding later data. Every nnterp value on this engine
is a copy, handed back to the model when the block moves on, so a saved value stays what it
was read as and an in-place edit still lands. Two reads of a value in one statement are the
same tensor:

```python
with model.trace("The Eiffel Tower is in the city of", temperature=0.0, max_tokens=1):
    layer.layer_output[:, -1] += 0.1 * layer.layer_output[:, -1]
    nudged = model.logits.save()

assert not torch.equal(nudged, logits)
```

### The stream is in two halves on most families

vLLM's Llama-style blocks fuse each residual add into the following norm: the block is called
`forward(positions, hidden_states, residual)` and returns `(hidden_states, residual)`. Natively
`layers[i].input` is the positions and `layers[i].output` a pair whose sum is the stream. Use
`layer_input` and `layer_output`; they are the stream on every family and both engines
(`layer_input` exists on `StandardizedTransformer` too, where it is `layers[i].input`).

### `logits` is one position

The engine computes logits for the last position only: `model.logits` is `[1, 1, vocab]`, the
prompt's last token on the prefill and the newest token on each decode step.
`model.samples` (nnsight's) is the token drawn from it.

### A written value must keep its rows

An assignment is spliced into the step the engine is running among other requests' rows. nnterp
refuses a value whose shape is not the one it served, in the block that wrote it; the request
fails and the engine carries on:

```python
try:
    with model.trace("The Eiffel Tower is in the city of", temperature=0.0, max_tokens=1):
        layer.layer_output = layer.layer_output[:, :-1]
except RuntimeError as error:
    print(str(error).splitlines()[0][:90])
```

Errors raised in the engine's worker arrive as `RuntimeError` naming the original type.

### The attention interior: what goes into the kernel and what comes out

vLLM's attention module projects (and rotates) its queries, keys and values and hands them to
the engine's attention layer, which returns the per-head outputs. Those four are served in
nnterp's layouts, heads split out, and are writable like any other value:

```python
attention = model.layers[12].self_attn
with model.trace("The Eiffel Tower is in the city of", temperature=0.0, max_tokens=1):
    queries = attention.attention_queries.save()         # [1, heads, tokens, head_dim]
    keys = attention.attention_keys.save()               # [1, kv_heads, tokens, head_dim]
    values = attention.attention_values.save()           # [1, kv_heads, tokens, head_dim]
    attention.attention_head_outputs[:, :, 0] = 0        # [1, tokens, heads, head_dim]: ablate head 0
    ablated = model.logits.save()

print(tuple(queries.shape), tuple(keys.shape), tuple(values.shape))
# (1, 9, 9, 64) (1, 3, 9, 64) (1, 3, 9, 64)
assert not torch.equal(ablated, logits)
```

They are this step's rows. On a decode step the keys and values are the new token's alone;
the earlier ones are in the engine's cache, which the kernel reads for itself (transformers'
eager attention is handed the whole cache). The queries and keys are read after the rotary
embedding, as on transformers.

### The pattern is recomputed: read-only, and the prefill's

The scores and the pattern are computed inside vLLM's attention kernel, which serves nothing
between its inputs and its output, and no attention backend of vLLM's does it in Python. nnterp
recomputes them from the queries and keys it serves, with the layer's own scale, softcap and
sliding window, in transformers' layout:

```python
with model.trace("The Eiffel Tower is in the city of", temperature=0.0, max_tokens=1):
    scores = attention.attention_scores.save()             # [1, heads, query, key]; -inf where a key is not seen
    pattern = attention.attention_probabilities.save()     # [1, heads, query, key]

print(tuple(pattern.shape), pattern[0, :, -1].sum(-1).mean().item())
# (1, 9, 9, 9) 1.0
```

Two things follow from "recomputed":

- **Read-only.** The kernel never takes a pattern, so an edit could not reach the model, and an
  assignment raises. To change what a head attends to, edit its queries or keys; to change what
  it wrote, edit its head outputs.
- **The prefill only.** A decode step holds one token's keys; the rest are in vLLM's cache.
  nnterp knows which step a read would be served before reading anything, and raises
  `nnterp.Unavailable` there (it arrives as a `RuntimeError` naming it, like any error from the
  engine's worker):

```python
try:
    with model.trace("The Eiffel Tower is in the city of", temperature=0.0, max_tokens=2, ignore_eos=True) as tracer:
        for step in tracer.iter[:2]:
            pattern = attention.attention_probabilities.save()
except RuntimeError as error:
    print("on a decode step" in str(error))
# True
```

A trace with no `tracer.iter` runs its block on the prefill, so the pattern is there whatever
`max_tokens` is. `support()`, which runs outside any step, lists both as available.

### What is unavailable

```python
support = model.support()
print(sorted(name for name, reason in support.items() if reason))
# ['attention_mask']
```

A request is one unpadded sequence, so there is no `attention_mask`. `input_ids` and `input_size` are read-only. `project_on_vocab`
and `get_topk_closest_tokens` call the engine's modules, so they work inside a trace only.
Gradients, `.source` inside a kernel and `scan` are nnsight's limits on this engine.

### `skip_layers` on a shared engine

A skip has to answer for every row of the step, and a step holds whatever requests the engine
batched together. `skip_layers` is safe on a request that runs alone; with other requests in
flight it can end the engine (nnsight's vLLM guide, "skip"). To ablate a block without skipping
it, write its contributions to zero.

## Family notes

What vLLM's block does decides which base a family uses; the last column is what differs from
the same family on transformers.

| family | vLLM's block | notes |
| --- | --- | --- |
| `llama`, `mistral`, `phi3`, `qwen2`, `qwen3` | fused: `(hidden_states, residual)` | names are transformers'. vLLM runs Phi-3 through its Llama classes. |
| `gemma` | fused | no `lm_head` module: the unembedding is `embed_tokens`' weight, and `project_on_vocab` uses it. `token_embeddings` is the lookup *before* the `sqrt(hidden_size)` scaling (transformers' module scales itself); `layers[0].layer_input` is the scaled stream. |
| `gemma2`, `gemma3_text` | fused, sandwich norms | contributions are the post-norms' outputs, as on transformers. `token_embeddings` as on `gemma`. `gemma2` has no `lm_head` module either. |
| `mixtral`, `qwen2_moe`, `qwen3_moe`, `olmoe` | fused, a mixture of experts for an MLP | `mlp_output` is the mixture's output. The routing is inside vLLM's fused MoE kernel: no router values. Mixtral's `block_sparse_moe` is aliased to `mlp`. |
| `deepseek_v2`, `deepseek_v3` | fused, latent attention, a mixture of experts | the attention's interior is unavailable: vLLM runs latent attention in its own kernel on compressed keys and values. That kernel is half-precision only; `VLLM_MLA_DISABLE=1` selects vLLM's ordinary attention layer, which runs in float32. |
| `exaone4` | returns `(stream, residual)`, post-norm | not fused, whatever the signature says: the first element is the whole stream. Contributions are the post-norms' outputs, as on transformers. |
| `cohere`, `cohere2` | returns `(stream, residual)`, parallel | not fused either. No `lm_head` module; `project_on_vocab` unembeds with `embed_tokens`. |
| `olmo3` | plain, positions first, post-norm | contributions are the post-norms' outputs, as on transformers. |
| `gpt2` | plain: takes and returns the stream | names are transformers' (`transformer.h`, `attn`, `ln_1`, `ln_2`) |
| `gptj`, `phi` | plain, positions first, parallel | the head has a bias, which `project_on_vocab` adds as the model does. |
| `gpt_neox` | plain, positions first | `embed_out` is `lm_head`. |
| `bloom`, `mpt` | plain, positions first, ALiBi | vLLM's attention and MLP modules return their contributions alone (transformers' add the residual inside). The recomputed scores carry the ALiBi bias as slope times distance behind the query, which differs from transformers' by one constant per query and leaves the pattern the same. Needs nnsight with the meta-build `tolist` fix. |
| `falcon` | plain, positions first, parallel | attention and MLP return `(output, bias)`. On a checkpoint whose projections have a bias (Falcon-RW) the contributions are unavailable. |

vLLM's attention kernels need a head width of at least 16 (32 for the float32 one), so the
tiny random checkpoints nnterp's transformers suite pins do not run on this engine; the vLLM
suite names a small real checkpoint per family.

## Adding a vLLM family

One module, `nnterp/families/vllm/<model_type>.py`, with what a transformers family declares
([adding-a-family](../extending/adding-a-family.md)): `RENAME`, `Layer` /
`Attention` / `Mlp`, and `ENVOYS` keyed on vLLM's module classes
(`vllm.model_executor.models.<name>`). Subclass the bases in `nnterp.components.vllm`:

- `FusedLayer` for a block that takes and returns `(hidden_states, residual)` whose sum is the
  stream; set `HIDDEN` / `RESIDUAL` if the block orders its arguments another way.
- `Layer` for a block that takes and returns the stream. `STREAM` is the stream's index in the
  block's call (`0` on GPT-2, `1` where the positions come first), and `returns_tuple = True`
  says the block returns the stream with something beside it (`exaone4`, `cohere`).
- `Attention` and `Mlp` when the module's output is what the block adds; otherwise point the
  value at the right place with `Flat`, the descriptor for a `[tokens, ...]` tensor
  (`Flat("../post_attention_layernorm.output")` in `gemma2.py`).
- `Attention`'s interior is the inputs and output of the module's `attn` child (vLLM's attention
  layer, called `self.attn(q, k, v)`), split into heads by that layer's `head_size`. A family
  whose attention names the child another way redefines the four values
  (`Flat("<child>.inputs", select=0, heads="first", ...)`); one with no such call marks them
  `unavailable(...)` (`deepseek_v2.py`).
- `def project_on_vocab(model, hidden)` where the model does not unembed with `lm_head` alone:
  `project(model, hidden, model.embed_tokens)` for tied embeddings (`gemma.py`),
  `project(model, hidden, model.lm_head, model.lm_head.bias)` for a head with a bias (`phi.py`).
- A size the config spells its own way is the transformers family's function, imported
  (`from ..falcon import num_kv_heads`).

Read vLLM's forward before choosing: the return type does not tell. Exaone4's and Cohere's
blocks return a pair whose first element is already the whole stream. Then add
`tests/vllm_families/test_vllm_<model_type>.py`, a `VLLMFamilySuite` subclass naming a
checkpoint both engines load, with an attention head at least 32 wide; the suite compares
every value with `StandardizedTransformer`'s.
`nnterp.families.register(module, engine="vllm")` (or `register(family, "my_type", engine="vllm")`) adds one from outside the package.
`StandardizedVLLM(repo_id, family=my_family)` uses one for a single load instead: the
lookup is skipped (the config is not read for it, nothing is registered), `model.family` is
`my_family`, and `rename=` / `envoys=` still layer on top. `my_family` is a module or a
`types.SimpleNamespace` with `RENAME` and `ENVOYS` keyed on vLLM's classes, so a shipped
family extends as on transformers
([registering](../extending/registering.md#passing-a-family-at-load)):

```python
import types
from nnterp import StandardizedVLLM
from nnterp.families.vllm import llama

family = types.SimpleNamespace(**vars(llama))
family.RENAME = {**llama.RENAME, "mlp": ["mlp", "ffn"]}
model = StandardizedVLLM("HuggingFaceTB/SmolLM2-135M-Instruct", family=family, dispatch=True,
                         gpu_memory_utilization=0.2, max_model_len=1024)
assert model.family is family and model.layers[0].ffn is model.layers[0].mlp
```

## Gotchas

- **nnterp must be importable in the engine's worker process.** The block runs there against
  the family's envoy classes, by reference. An installed package is; a path added with
  `sys.path.insert` in the script is not, but `PYTHONPATH` is inherited.
- **Bind the envoys you use outside the block in a sweep.** `model.layers[i]` or `model.logits`
  inside a block ships the root with every invoke (nnsight's vLLM guide, "per-invoke cost").
- **`layers[i].input` is not the stream** on a fused family; it is the positions.
- **Reads follow forward order** within a step, as everywhere: a block's `layer_input`, its
  attention's queries, keys and values (and the scores and pattern, which are computed from
  them), then its head outputs, the contributions, then its `layer_output`.
- **Leave prefix caching off for `model.edit()`.** nnsight builds the engine with
  `enable_prefix_caching=False`; with it on, an edit sees only the uncached tail of a repeated
  prefix (nnsight warns). A trace recomputes its own prompt either way.
- **bf16 checkpoints differ from transformers by more than float32 ones**; the suite compares
  in float32 (`dtype="float32"`).
- **The two engines' attention kernels are not bit-identical.** Given the same queries, keys and
  values, the head outputs agree to about 1e-3 of their scale on most checkpoints and to 2e-2 on
  one with a very sharp softmax (Qwen2.5-0.5B's first block).

## Related

- nnsight `docs/models/vllm.md` and the `nnsight:vllm` skill — the engine: sampling, `edit()`,
  taps, serving, tensor parallelism.
- [residual-stream](residual-stream.md), [root-values](root-values.md), [availability](availability.md).
- [docs/developing/testing.md](../developing/testing.md) — running `tests/vllm_families/`.
