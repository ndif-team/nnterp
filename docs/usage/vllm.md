---
title: The vLLM engine
one_liner: "`StandardizedVLLM` is nnsight's `VLLM` with nnterp's names and values: the same layouts as `StandardizedTransformer` (batch axis 1), private copies, vLLM's own defaults, and a family per vLLM implementation under `nnterp/families/vllm/`."
tags: [usage, vllm, engine, StandardizedVLLM, layer_input, families]
related: [docs/usage/loading.md, docs/usage/residual-stream.md, docs/usage/root-values.md, docs/usage/availability.md, docs/usage/generation.md, docs/extending/adding-a-family.md]
sources: [nnterp/standardized_vllm.py, nnterp/components/vllm/__init__.py, nnterp/components/vllm/flat.py, nnterp/components/vllm/layer.py, nnterp/components/vllm/attention.py, nnterp/components/vllm/transformers_backend.py, nnterp/families/vllm/__init__.py, nnterp/families/vllm/llama.py, nnterp/families/vllm/gemma2.py, nnterp/families/vllm/gpt2.py, tests/vllm_families/vllm_suite.py]
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

Families on this engine (60): `llama`, `mistral`, `ministral3`, `phi3`, `qwen2`, `qwen3`,
`ernie4_5`, `glm`, `arcee`, `apertus`, `seed_oss`, `nemotron`, `gemma`, `gemma2`, `gemma3_text`,
`exaone4`, `hyperclovax`, `cohere`, `cohere2`, `granite`, `granitemoe`, `stablelm`,
`hunyuan_v1_dense`, `hunyuan_v1_moe`, `mixtral`, `phimoe`, `qwen2_moe`, `qwen3_moe`, `olmoe`,
`flex_olmo`, `ernie4_5_moe`, `glm4_moe`, `glm4_moe_lite`, `minimax_m2`, `afmoe`, `laguna`,
`llama4_text`, `mimo_v2_flash`, `gpt_oss`, `dbrx`, `deepseek_v2`, `deepseek_v3`, `kimi_k2`,
`qwen2_vl_text`, `qwen2_5_vl_text`, `qwen3_vl_text`, `gpt2`, `gptj`, `gpt_neox`, `bloom`, `mpt`,
`falcon`, `phi`, `opt`, `olmo3`, and on vLLM's transformers backend `olmo`, `olmo2`, `smollm3`,
`starcoder2`, `gpt_bigcode` ([family notes](#family-notes)). Another `model_type` raises `UnsupportedFamily`.

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
| `llama`, `mistral`, `phi3`, `qwen2`, `qwen3`, `ernie4_5`, `arcee`, `seed_oss`, `nemotron` | fused: `(hidden_states, residual)` | names are transformers'. vLLM runs Phi-3 and ERNIE-4.5 through its Llama classes. |
| `glm` | fused (vLLM's Llama classes) | vLLM sets `partial_rotary_factor` to 0.5 on every GLM checkpoint whatever its config says ([differences](#known-differences-between-the-engines)). |
| `ministral3` | fused (vLLM's `mistral` module) | an HF-format checkpoint needs `hf_overrides={"rope_parameters": {..., "apply_yarn_scaling": False}}`, or every query and key is 1.28 times transformers' ([differences](#known-differences-between-the-engines)). |
| `apertus` | fused | transformers' norm names, `attention_layernorm` and `feedforward_layernorm`, are aliased to `input_layernorm` and `post_attention_layernorm`. Building the model needs an nnsight whose meta build answers `.cpu().item()` (vLLM's xIELU reads its scalars at construction). |
| `hyperclovax` | returns `(stream, residual)`, sandwich norms, Granite's multipliers | not fused: the block ignores its `residual` argument and returns the whole stream first. Contributions are the post-norms' outputs times `residual_multiplier` (the module's own where `use_post_norm` is off); `token_embeddings` precedes `embedding_multiplier`; `logits_scaling` is in the logits processor. |
| `granite`, `granitemoe` | plain, positions first | contributions are the module's output times `residual_multiplier`, a computed copy divided back on a write (a `Flat` with a `factor`), as on transformers. `token_embeddings` precedes `embedding_multiplier`; vLLM's logits processor divides by `logits_scaling`. GraniteMoE's `block_sparse_moe` is `mlp`. |
| `stablelm` | returns `(stream, stale residual)` | not fused: the first element is the whole stream, and the second is not a half of it, so summing the pair is wrong. |
| `hunyuan_v1_dense`, `hunyuan_v1_moe` | fused, plus a third element | the block returns `(hidden_states, residual, kv_states)` and the attention `(output, (k, v))`; the third element (keys and values for cross-layer sharing) passes through edits untouched. The cross-layer attention class (`use_cla`) is not mapped. |
| `gemma` | fused | no `lm_head` module: the unembedding is `embed_tokens`' weight, and `project_on_vocab` uses it. `token_embeddings` is the lookup *before* the `sqrt(hidden_size)` scaling (transformers' module scales itself); `layers[0].layer_input` is the scaled stream. |
| `gemma2`, `gemma3_text` | fused, sandwich norms | contributions are the post-norms' outputs, as on transformers. `token_embeddings` as on `gemma`. `gemma2` has no `lm_head` module either. |
| `mixtral`, `qwen2_moe`, `qwen3_moe`, `olmoe`, `ernie4_5_moe`, `glm4_moe`, `minimax_m2` | fused, a mixture of experts for an MLP | `mlp_output` is the mixture's output. The routing is inside vLLM's fused MoE kernel: no router values. Mixtral's and MiniMax-M2's `block_sparse_moe` is aliased to `mlp`. |
| `phimoe` | returns `(stream, residual)`, a mixture of experts | not fused: the first element is the whole stream. `block_sparse_moe` is `mlp`. vLLM's logits leave out `lm_head.bias` ([differences](#known-differences-between-the-engines)). |
| `afmoe`, `laguna` | fused, a mixture with a shared expert | the shared expert serves no `mlp_output` of its own (it is inside the mixture's output). AFMoE's contributions are the post-norms' outputs, as on transformers. Laguna's `attention_output` is the gated projection, its head outputs ungated, and a block's head counts are its own attention layer's. |
| `llama4_text` | fused, dense and mixture blocks interleaved, chunked attention | queries and keys are served in transformers' channel order (vLLM permutes the projections' rows at load, and the values undo it). A chunked block's recomputed pattern masks other chunks. `feed_forward` is `mlp`. |
| `mimo_v2_flash` | fused | values and head outputs are `v_head_dim` wide, queries and keys `head_dim`; sliding blocks have their own kv-head count and a sink, so their recomputed pattern's rows sum to less than one, as on transformers. |
| `gpt_oss` | fused, the stream first: `(hidden_states, positions, residual)` | `model.embedding` is `embed_tokens` and `attn` is `self_attn`. The recomputed pattern includes each head's sink column and drops it, so rows sum to less than one, as on transformers. |
| `flex_olmo` | returns `(stream, None)`, post-norm | not fused: the first element is the whole stream. Contributions are the post-norms' outputs, as on transformers. |
| `dbrx` | plain, positions first | the attention sits inside `norm_attn_norm`; `ffn` is `mlp`. vLLM 0.27.1 cannot load DBRX weights (its loader looks for the experts where the mixture no longer keeps them); the suite's class skips there. |
| `deepseek_v2`, `deepseek_v3` | fused, latent attention, a mixture of experts | the attention's interior is unavailable: vLLM runs latent attention in its own kernel on compressed keys and values. That kernel is half-precision only; `VLLM_MLA_DISABLE=1` selects vLLM's ordinary attention layer, which runs in float32. |
| `glm4_moe_lite` | fused, latent attention, a mixture of experts | as `deepseek_v2` (it is vLLM's DeepSeek-V2 attention). vLLM's latent norms use the config's `rms_norm_eps` where transformers' use 1e-6 ([differences](#known-differences-between-the-engines)). |
| `kimi_k2` | DeepSeek-V3's, inside the Kimi-K2.5 wrapper at `language_model.model` | as `deepseek_v3`; text only: the engine needs `language_model_only=True`. |
| `qwen2_vl_text`, `qwen2_5_vl_text`, `qwen3_vl_text` | Qwen2's or Qwen3's fused block, inside the VL wrapper at `language_model.model` | text only: the engine needs `language_model_only=True` (in its multimodal mode vLLM embeds the prompt outside the forward and the model never sees the ids). On text the M-RoPE positions are three identical rows and Qwen3-VL adds no deepstack features. The vision tower is not mapped on this engine. |
| `exaone4` | returns `(stream, residual)`, post-norm | not fused, whatever the signature says: the first element is the whole stream. Contributions are the post-norms' outputs, as on transformers. |
| `cohere`, `cohere2` | returns `(stream, residual)`, parallel | not fused either. No `lm_head` module; `project_on_vocab` unembeds with `embed_tokens`. |
| `olmo3` | plain, positions first, post-norm | contributions are the post-norms' outputs, as on transformers. |
| `gpt2` | plain: takes and returns the stream | names are transformers' (`transformer.h`, `attn`, `ln_1`, `ln_2`) |
| `gptj`, `phi` | plain, positions first, parallel | the head has a bias, which `project_on_vocab` adds as the model does. |
| `gpt_neox` | plain, positions first | `embed_out` is `lm_head`. |
| `opt` | plain: takes and returns the stream | no `mlp` on either engine, as on transformers (`fc1`/`fc2` sit on the block). `attention_queries` are served times `head_dim**-0.5`, as transformers scales them. With tied embeddings `lm_head` is `embed_tokens`. |
| `bloom`, `mpt` | plain, positions first, ALiBi | vLLM's attention and MLP modules return their contributions alone (transformers' add the residual inside). The recomputed scores carry the ALiBi bias as slope times distance behind the query, which differs from transformers' by one constant per query and leaves the pattern the same. Needs nnsight with the meta-build `tolist` fix. |
| `falcon` | plain, positions first, parallel | attention and MLP return `(output, bias)`. On a checkpoint whose projections have a bias (Falcon-RW) the contributions are unavailable. |
| `olmo`, `olmo2`, `smollm3`, `starcoder2`, `gpt_bigcode` | transformers' own block, on vLLM's transformers backend | vLLM has no implementation of its own and runs transformers' modules (`nnterp.components.vllm.transformers_backend`): activations carry the batch axis inside the model, and the engine's attention layer, which the backend keeps outside the module tree, is mounted as `self_attn.attn` inside a trace. OLMo-2's contributions are the post-norms' outputs. GPT-BigCode's tree is `model.{wte, h, ln_f}`. `.source` on these modules does not work: the backend rewrites their forwards. |

vLLM's attention kernels need a head width of at least 16 (32 for the float32 one; a narrower
one can hang the engine rather than fail), and most of its MLPs take SiLU only, so most of the
tiny random checkpoints nnterp's transformers suite pins do not run on this engine. The vLLM
suite names a small real checkpoint where one is published, and otherwise a tiny one with heads
at least 32 wide, or a copy of one resized for vLLM, written once by the test.

### Known differences between the engines

vLLM's implementation of a family is not always transformers' model, and transformers' is not
always the checkpoint's. Where the two disagree nnterp serves what vLLM computes; the suite
aligns the two engines for the comparison and says how in each test file. Found while adding
the families above, on vLLM 0.27.1 and transformers 5.15:

- **GLM** (`GlmForCausalLM`): vLLM sets `partial_rotary_factor` to 0.5 whatever the config says.
  GLM-Edge says 1.0, so vLLM rotates half of each head where transformers rotates all of it.
- **Ministral 3**: vLLM's plain YaRN rotary ignores `mscale` and `mscale_all_dim` and multiplies
  cos and sin by `0.1 * ln(factor) + 1` (1.277 at factor 16), which transformers' equal
  `mscale`/`mscale_all_dim` cancel. Mistral's own `params.json` (`yarn.apply_scale: false`) turns
  it off; an HF-format checkpoint needs
  `hf_overrides={"rope_parameters": {..., "apply_yarn_scaling": False}}`.
- **GLM-4.7-Flash** (`glm4_moe_lite`): vLLM's latent-attention norms (`q_a_layernorm`,
  `kv_a_layernorm`) use the config's `rms_norm_eps` (1e-5); transformers fixes 1e-6. DeepSeek's
  configs say 1e-6, so only the GLM checkpoints differ.
- **GLM-4.5 MoE** (`glm4_moe`): vLLM ignores `tie_word_embeddings`; on a tied checkpoint its
  `lm_head` is never loaded and the logits are garbage.
- **PhiMoE**: vLLM loads `lm_head.bias` but its logits processor is called without it, so the
  logits lack the bias transformers adds.
- **StableLM**: vLLM builds the sequential block only, with no query/key norms:
  `use_parallel_residual` and `qk_layernorm` are ignored, so on StableLM-2-12B (both set) it runs
  another model.
- **FlexOlmo**: vLLM's MLP is SiLU whatever `hidden_act` says.
- **Llama 4**: vLLM permutes the rows of `q_proj`/`k_proj` at load (its rotary pairs channels
  differently); nnterp's queries and keys undo it. vLLM reads `attn_temperature_tuning` from
  the generation config, not the model config, and by default turns it on only above 32K
  positions; pass `override_generation_config={"attn_temperature_tuning": True}` to match
  transformers on NoPE blocks past position 8191.
- **DBRX**: transformers' `DbrxConfig` ignores `attn_config.rope_theta` and rotates with 10000
  unless `rope_parameters` is given, where vLLM uses the checkpoint's 500000. And vLLM 0.27.1's
  `DbrxModel.load_weights` looks for the experts under `ffn.experts` where its mixture keeps them
  under `ffn.experts.routed_experts`, so it loads no DBRX checkpoint.
- **Kernels**: vLLM's float32 attention kernels differ from an exact softmax by about 1e-3 of the
  head outputs' scale (up to 7e-3 on a sharp softmax) even where the queries, keys and values agree
  to 1e-7; a few test files set `KERNEL_TOLERANCE` (and `TOLERANCE`) with the measurement beside it.

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
- A value that is a tensor times a scalar of the model is a `Flat` with a `factor`, a function
  of the host giving the scalar: it is served multiplied and divided back on a write
  (`granite.py`'s contributions, `opt.py`'s queries).
- A family vLLM runs through its transformers backend (transformers' own modules) builds on
  `nnterp.components.vllm.transformers_backend` (`olmo2.py`): its `Layer`, `Attention` and `Mlp`
  serve tensors that already carry the batch axis (`Flat(..., batch=True)`) and mount the
  engine's attention layer as `self_attn.attn`.

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
