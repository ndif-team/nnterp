---
title: Glossary
one_liner: Alphabetical definitions of the terms the nnterp docs use, one or two sentences each, with the page that explains each one.
tags: [reference, glossary, vocabulary]
related: [docs/reference/api-quick-reference.md, docs/reference/families.md, docs/usage/vocabulary.md, docs/usage/residual-stream.md, docs/usage/availability.md, docs/usage/layouts.md, docs/usage/attention-interior.md, docs/usage/delta-net.md, docs/extending/adding-a-family.md, docs/developing/eproperty-internals.md, docs/developing/recurrent-mixer-internals.md]
sources: [nnterp/__init__.py, nnterp/standardized.py, nnterp/components/__init__.py, nnterp/components/eproperty.py, nnterp/components/attention.py, nnterp/components/linear_attention.py, nnterp/components/recurrent.py, nnterp/families/__init__.py, tests/families/suite.py]
---

# Glossary

## What this is for

The nnterp docs use each of these words in exactly one sense. This page gives that sense in one or two sentences and points at the page that explains it; nnsight's own terms (envoy, eproperty, occurrence, source, `tracer.iter`) are defined here only as far as nnterp leans on them, and nnsight `docs/reference/glossary.md` has the rest.

## Canonical pattern

The terms in one block: a *family* resolved from the checkpoint, a *standard name* aliasing a *native name*, *availability* reported by `support()` and enforced by `Unavailable`.

```python
from nnterp import StandardizedTransformer, Unavailable

model = StandardizedTransformer("meta-llama/Llama-3.1-8B", dispatch=True)   # the checkpoint's own attention: sdpa

print(model.family.__name__)                                                # nnterp.families.llama
print(model.layers[0].self_attn is model.model.layers[0].self_attn)         # True: the standard name is an alias of the native one
print(model.support()["self_attn.attention_probabilities"][0])              # read inside the eager attention forward, but this model runs 'sdpa'; ...

try:
    with model.trace("Hello"):
        model.layers[0].self_attn.attention_probabilities.save()
except Unavailable as error:
    print(error)                                                            # raised at that line, before the model runs
```

## Alias

An extra attribute nnsight's `rename=` installs on an envoy so it answers to a second name; the same object, not a copy, so the native name keeps working and iteration does not double-count. Every standard name is one. See [../usage/vocabulary.md](../usage/vocabulary.md) and nnsight `docs/usage/rename-modules.md`.

## Attention sink

A learned per-head logit that joins the softmax as one extra key column and is dropped afterwards (GPT-OSS). The pattern's rows then sum to less than one, and `attention_scores` is read just before the sink column joins. See [families.md](families.md#attention-sink) and [../usage/attention-interior.md](../usage/attention-interior.md).

## Availability, `Unavailable`

Whether this checkpoint has a standard value. `support()` returns `None` (available) or a reason string; reading or writing an unavailable value raises `nnterp.Unavailable` with the same reason, at that line, before the model runs. Decided per envoy by the descriptor's `unavailable=` (a string, or a predicate of the envoy), so a config flag or a hybrid's block type can decide. See [../usage/availability.md](../usage/availability.md).

## Chunked kernel, recurrent kernel

The two delta-rule kernels a gated DeltaNet forward can run: `torch_chunk_gated_delta_rule` on a prompt (the state carried between 64-token chunks) and `torch_recurrent_gated_delta_rule` on a decode step (one token at a time). The two compute the same rule; only the recurrent one materializes the state after every token. See [../usage/delta-net.md](../usage/delta-net.md).

## Contribution

What a sublayer adds to the residual stream: `attention_output` and `mlp_output`, defined by the identity `layers[i].input + attention_output + mlp_output == layer_output`, which holds on sequential, parallel and sandwich blocks alike because each family points the value at the tensor that is added. See [../usage/residual-stream.md](../usage/residual-stream.md).

## Eager attention

transformers' `attn_implementation="eager"`: the Python attention forward whose operations `.source` can read, as opposed to `sdpa` or flash attention, which compute the same thing inside one opaque call. The interior attention values need it; the constructor does not force it. See [../usage/loading.md](../usage/loading.md) and [../usage/attention-interior.md](../usage/attention-interior.md).

## Envoy

nnsight's proxy for one module in the tree, reached by attribute path (`model.layers[3].self_attn`), with `.input`, `.output`, `.source`, `.skip()`. nnterp's `Layer`, `Attention`, `Mlp`, `RecurrentMixer` and `LinearAttention` are `Envoy` subclasses installed through `envoys=`. See nnsight `docs/reference/glossary.md`.

## `envoys=`

The `TransformersModel` argument mapping module types (or path suffixes, native or aliased) to `Envoy` subclasses; a family's `ENVOYS` fills it and a user's `envoys=` merges on top, a key given there winning. nnsight tries type keys before path keys. See [../extending/custom-values.md](../extending/custom-values.md).

## eproperty

nnsight's descriptor for a served value: reading it parks the intervention until the model reaches its location, writing it replaces the value there, and it appears in the envoy's repr with its description. nnterp's `EProperty` adds availability and a path for a key (`output`, `../norm.output`, `source.<op>.inputs`), so one descriptor serves a value wherever it lives; `DerivedEProperty` computes one instead. See [../developing/eproperty-internals.md](../developing/eproperty-internals.md) and nnsight `docs/developing/extending-envoy.md`.

## Family

One module under `nnterp/families/`, named after `config.model_type` (`gpt2.py`, `gemma3_text.py`), declaring `RENAME`, its `Layer` / `Attention` / `Mlp` (and `LinearAttention`) subclasses and `ENVOYS`. `model.family` is the one a checkpoint resolved to. See [families.md](families.md) and [../extending/adding-a-family.md](../extending/adding-a-family.md).

## Gated DeltaNet

The linear-attention mixer of Qwen3-Next and Qwen3.5 (`linear_attn`): queries, keys and values projected like attention, but mixed through a per-head recurrent state that each token decays by a gate (`decays`), writes into with a strength (`betas`), and that the query reads against. No pattern and no scores. See [../usage/delta-net.md](../usage/delta-net.md).

## Hybrid

A family whose blocks are of two kinds: Qwen3-Next, Qwen3.5 (text), Qwen3.5-MoE (text), OLMo-Hybrid and Kimi-Linear have `linear_attn` (gated DeltaNet, a `LinearAttention`, a `RecurrentMixer`) on three blocks in four and `self_attn` on the fourth, per `config.layer_types`; never both on one block. `support()` reads per block there. See [families.md](families.md#hybrids).

## Image features, image token mask

The two values of a vision tower that say where the image enters the text model: `vision.image_token_mask` is `input_ids == image_token_id` (`[batch, seq]`, read off the model's inputs, so first in a trace), and `vision.image_features` is the tensor the wrapper scatters into the token embeddings at those positions (`[image_tokens, hidden]`, read at the scatter, assignable). `layers[0].input[vision.image_token_mask] == vision.image_features` on every wrapper. See [../usage/vision.md](../usage/vision.md#where-the-image-meets-the-text-model).

## Interface (`attention_interface_1`)

transformers' shared attention call, `attention_interface(module, query, key, value, attention_mask, ...)`, which under eager attention is `eager_attention_forward`. The base `Attention` reads the interior values on and inside this call; four families do their own arithmetic and relocate them. See [../usage/attention-interior.md](../usage/attention-interior.md).

## KV heads, grouped-query attention

The number of key/value heads, `num_kv_heads`, fewer than `num_heads` under grouped-query attention and 1 under multi-query. `attention_keys` and `attention_values` are read before `repeat_kv`, so their head axis is `kv_heads` wide, except where the family expands them first (DeepSeek, Falcon's 40B layout). See [../usage/layouts.md](../usage/layouts.md).

## Layout, dims

The shape a value has on every family, as a `jaxtyping` annotation on the descriptor: one of thirty-five named aliases, each defined beside the envoy that serves it (`Residual = Float[Tensor, "batch seq hidden"]` in `components/layer.py`, `Pattern` and `Keys` in `components/attention.py`, `State` in `components/linear_attention.py`, `Logits` in `standardized.py`, ...); `nnterp.components` re-exports the thirty-two envoy-level names. `value.layout` is that alias itself (`Attention.attention_keys.layout is Keys`, usable with `isinstance`), `value.dims` the axis names as a tuple. A family's redefinition and a custom value annotate with the same name. Layouts differ between values, not between families, except `layer_output` on DeepSeek-V4 (`Streams`) and `linear_attn.decays` on Kimi-Linear (`ChannelGates`). See [../usage/layouts.md](../usage/layouts.md).

## MLA (multi-head latent attention)

DeepSeek-V2/V3's attention, where queries and keys are `qk_head_dim = qk_nope_head_dim + qk_rope_head_dim` wide, values `v_head_dim`, and the interface sees `num_heads` key/value heads. The root publishes `qk_head_dim` beside `head_dim` for it. See [families.md](families.md#latent-attention).

## Native name

The module's name in the checkpoint's own architecture (`transformer.h[i].attn` on GPT-2, `gpt_neox.layers[i].attention` on Pythia). It keeps working under nnterp; `RENAME` maps it onto the standard name, and `envoys=` path keys match it (and an alias path where the alias's `rename` key ends it). See [../usage/vocabulary.md](../usage/vocabulary.md).

## Occurrence

nnsight's count of how many times a run has reached one location; each visit is an occurrence, and `tracer.iter[t]` binds a read to occurrence `t`. Occurrences are counted per location over the whole run, so a decode step's token is not occurrence 0 once an earlier step fired the same op. See [../usage/delta-net.md](../usage/delta-net.md) and nnsight `docs/usage/iter-all-next.md`.

## Operation, op

One call or assignment inside a module's forward as nnsight's `.source` names it: `nn_functional_softmax_0`, `attn_weights_1`, `torch_chunk_gated_delta_rule_0`; `<callable>_<n>` for the n-th call, `<name>_<n>` for the n-th binding. An `EProperty`'s key names one after a `source` segment, with another `source` between a call and an op inside it (`source.attention_interface_1.source.nn_functional_softmax_0.output`). See [../extending/finding-source-ops.md](../extending/finding-source-ops.md) and nnsight `docs/usage/source.md`.

## Packed tower

A vision tower that runs every image of the invoke as one sequence of patches (the Qwen ViT, Pixtral): its `Patches` values are `[1, all patches, vision_hidden]`, with the images axis 1, and the processor's per-image grid (`image_grid_thw`, `image_sizes`) says where one image's patches end. See [../usage/vision.md](../usage/vision.md#the-qwen-vit).

## Parallel block

A block where one norm's output feeds both sublayers and `x + attn(norm(x)) + mlp(norm(x))` is summed at the end: GPT-NeoX (with `use_parallel_residual`), Phi, GPT-J, CodeGen, StableLM-2, Falcon. `mlp.input` is that norm's output, and the contribution identity holds unchanged. See [families.md](families.md#parallel-blocks).

## Pinned read, relaxed read

Inside `for t in tracer.iter[t]:` a read is *pinned* to occurrence `t` of its location; a read outside any `tracer.iter`, or after a step body's first read, is *relaxed* and takes the occurrence in flight. A `RecurrentMixer`'s per-token `state` decides its kernel and drills into the kernel call relaxed (`pinned(None)`), so the callee resolves from the live call, and the value read that follows is pinned to a token. See [../developing/eproperty-internals.md](../developing/eproperty-internals.md) and [../developing/recurrent-mixer-internals.md](../developing/recurrent-mixer-internals.md).

## Projector

`model.projector`: the last module before the scatter on a vision-language wrapper (`multi_modal_projector`, Qwen's `merger`, Idefics 3's `connector`), mapping the tower's output into the text model's width. `projector.input` is what the tower hands over; `projector.output` is `image_features` on most wrappers but not all (LLaVA-NeXT adds newline tokens after it), which is why `image_features` is read at the scatter instead. See [../usage/vision.md](../usage/vision.md#where-the-image-meets-the-text-model).

## Recurrent state

A gated DeltaNet layer's per-head memory, `[batch, heads, key_dim, value_dim]`: `state_input` entering a call (`None` on a fresh prompt), `state_output` leaving it, `state` after one token and `states` after every token of the call. See [../usage/delta-net.md](../usage/delta-net.md). A Mamba-1 layer's is one `state_dim` vector per channel, `[batch, channels, state_dim]`, under the same names ([../usage/selective-scan.md](../usage/selective-scan.md)).

## `RecurrentMixer`

The base envoy of a recurrent mixer (`nnterp.components.recurrent`): a subclass names its prompt and decode-step kernels (`CHUNK_KERNEL`, `RECURRENT_KERNEL`), the bindings the forward's test between them reads (`BRANCH`, `SEQ_OP`) and the per-token state binding (`STATE_OP`, or `None`), and declares its values at the kernel call; the base reaches them, reports their availability, holds `attention_output` and the per-token `state` / `states`, and routes the kernels (`route_kernels`). `LinearAttention` is the gated DeltaNet subclass, `SelectiveScan` the Mamba-1 one. See [../developing/recurrent-mixer-internals.md](../developing/recurrent-mixer-internals.md).

## Registry

`nnterp.families`: `lookup(model_type)` returns a family passed to `register()` (kept in `REGISTRY`), else imports `nnterp.families.<model_type>` on first use, else warns and returns `nnterp.families.default`, the best-effort family (whose load-time check raises `UnsupportedFamily` when it cannot standardize the checkpoint). The module names are the registry; `import nnterp` loads no transformers modeling module. See [../extending/registering.md](../extending/registering.md).

## Rename

nnsight's `rename=` constructor argument, a dict of native path to alias. A key with several components (`transformer.h`) binds where it resolves from, the root; a single-component key (`attn`) binds in every block that has one; a key that resolves nowhere is skipped. A family's `RENAME` is this dict. See nnsight `docs/usage/rename-modules.md`.

## Residual stream

The tensor a block passes to the next, `[batch, seq, hidden]`: `layers[i].input` entering, `layer_output` leaving, each sublayer adding its contribution in between. See [../usage/residual-stream.md](../usage/residual-stream.md).

## `route_kernels`, `route_delta_rule`

`nnterp.route_kernels(family, "torch" | "default")`: binds a family's recurrent kernel names, process-wide, to transformers' pure-torch kernels (on a gated DeltaNet, both to the token-by-token loop; on Mamba-1, each to its own, the scan being the loop) or back to the module's import-time binding. The per-token `state` / `states` exist only under `"torch"`; call it before tracing the layer. `nnterp.route_delta_rule(family, "recurrent" | "chunked")` is the same switch in the delta rule's words. See [../usage/delta-net.md](../usage/delta-net.md).

## Sandwich block

A block that norms a sublayer's output before adding it to the residual stream, `x + post_attention_layernorm(attn(...))` (Gemma-2/3/4, OLMo-2/3). The contribution is the post-norm's output, so those families point `attention_output` / `mlp_output` at the sibling norm. See [families.md](families.md#sandwich-norms) and [../extending/overriding-values.md](../extending/overriding-values.md).

## Scatter

The step in a vision-language wrapper's forward that writes the projected image features into the token embeddings at the image tokens (`masked_scatter`). `vision.image_features` is read there, through an `ImageScatter` envoy keyed on the wrapper's model, or on the root's forward where the family sets `ROOT_SCATTER` (Llama 4). See [../developing/vision-design.md](../developing/vision-design.md#the-image-values).

## Selective scan, `SelectiveScan`

Mamba-1's mixer: per channel, `h = exp(dt*A) h + dt B x`, `y = C.h + D x`, with the step size `dt` and the vectors `B`, `C` computed from each token. `nnterp.components.SelectiveScan` serves it as `linear_attn` on Mamba, Falcon-Mamba and Jamba, with `C` / `B` / `x` as `attention_queries` / `attention_keys` / `attention_values`, `dt` as `betas` and `dt*A` as `decays`. See [../usage/selective-scan.md](../usage/selective-scan.md).

## Softcap

Gemma-2's `final_logit_softcapping`: `cap * tanh(logits / cap)` applied after `lm_head`. `model.logits` and `project_on_vocab` include it; `model.lm_head.output` does not. See [../usage/root-values.md](../usage/root-values.md) and [../patterns/logit-lens.md](../patterns/logit-lens.md).

## Source-located value

A value whose `EProperty` path has a `source` segment (`inside_forward()`): a standard value read at an operation inside a forward through nnsight `.source`, drilled into before every read or write because an operation inside a called function exists only once someone has drilled into that call in the current run. `attention_probabilities` and the whole attention interior are ones. See [../extending/finding-source-ops.md](../extending/finding-source-ops.md) and nnsight `docs/usage/source.md`.

## Standard name

An alias from Llama's vocabulary that every family answers to: `embed_tokens`, `layers[i]`, `layers[i].self_attn`, `layers[i].mlp`, `norm`, `lm_head`, with `linear_attn` on a hybrid. Norm names are not part of the vocabulary; `input_layernorm` and `post_attention_layernorm` bind where a family spells them otherwise, but their meaning varies. See [../usage/vocabulary.md](../usage/vocabulary.md).

## Standard value

An `EProperty` on a `Layer`, `Attention`, `Mlp` or `LinearAttention` envoy or on the root, meaning the same thing on every family: `layer_output`, `attention_output`, `mlp_output`, `attention_probabilities`, `logits`, `states`, and the rest of [api-quick-reference.md](api-quick-reference.md). Listed in the repr, reported by `support()`, checked by the suite. See [../usage/root-values.md](../usage/root-values.md).

## `StandardizedProperty`

The descriptor each root size is (`num_layers`, `hidden_size`, `vocab_size`, `num_heads`, `num_kv_heads`, `head_dim`, `qk_head_dim`, `intermediate_size`): outside a trace, a plain rule over the config, unless the model's family module defines a function of the same name, which then answers (`falcon.num_kv_heads`, `deepseek_v2.head_dim`, `gpt2.intermediate_size`). Read-only: assigning one raises `AttributeError`. Not an eproperty: no location, nothing in the repr or `support()`. See [../usage/root-values.md](../usage/root-values.md#sizes).

## `support()`

`model.support()` (every value, `None` or `{layer: reason}`), `model.support(layer=i)` (one block, flat) and `envoy.support()` (one envoy): what this checkpoint has, computed from the tree and the config without running anything. See [../usage/availability.md](../usage/availability.md).

## Step (`tracer.iter`)

One iteration of nnsight's `for step in tracer.iter[...]:`: a decode step under `generate`, or on a DeltaNet layer routed recurrent, one token of the prompt when the loop walks `state`. A step body starts pinned to its step and relaxes after its first read. See [../usage/generation.md](../usage/generation.md) and nnsight `docs/usage/iter-all-next.md`.

## Sublayer

The attention (or linear attention) and the MLP of a block, each adding one contribution to the residual stream. `self_attn.input` and `mlp.input` are what enters each one, whatever norm produced it. See [../usage/residual-stream.md](../usage/residual-stream.md).

## Tiny checkpoint

The random-weights checkpoint each family's test pins (`hf-internal-testing/tiny-random-LlamaForCausalLM`, `yujiepan/qwen3.5-tiny-random`), small enough to run on a CPU in seconds; every example in these docs was run against one. Its outputs are meaningless as numbers: show shapes and structure from it, not values. See [families.md](families.md) and [../developing/testing.md](../developing/testing.md).

## Tower, `Vision`

The vision side of an image-text-to-text wrapper under its standard names: `model.vision` (a `Vision` envoy), `vision.layers[i]` (`VisionLayer`, with `VisionAttention` and `VisionMlp`), `vision.patch_embed`, `vision.norm`, the tower's sizes, `patch_embeddings` and `tower_output` (`Patches`, `[images, patches, vision_hidden]`), and the two image values. Named by the text model's family, since the family is one per `model_type` and the wrapper's `text_config.model_type` picks it. See [../usage/vision.md](../usage/vision.md#the-names).

## Tuple block, `returns_tuple`

A decoder block whose forward returns `(hidden_states, ...)` rather than the tensor alone: GPT-J, GPT-Neo, CodeGen, GPT-NeoX-Japanese, BLOOM, MPT, Falcon set `Layer.returns_tuple = True`. `layer_output` is the tensor either way; `skip_layers` packs the replacement the way the block would have. See [families.md](families.md#tuple-blocks).

## Gotchas

- "Standard name" is a name; "standard value" is a served tensor. `model.layers[3].self_attn` is a name; `model.layers[3].self_attn.attention_output` is a value.
- `post_attention_layernorm` is a standard *name* but not a standard *meaning*: on a sandwich block it follows the attention. What enters a sublayer is `self_attn.input` / `mlp.input`.
- "Available" is per envoy: on a hybrid the same value is available on some blocks and `"no self_attn module on this block"` on others.
- `hasattr(envoy, value)` raises `Unavailable` rather than returning `False`; `support()` is the question to ask.
- "Occurrence" counts over the whole run, "step" counts `tracer.iter` iterations; `states`, `state_after` and `set_state_after` translate between the two by counting from the current call's first token.
- A "wrapper" is the image-text-to-text class around a text model (`LlavaForConditionalGeneration` around a Llama); its family is the text model's, and `model.vision` exists only when it is loaded with `task="image-text-to-text"`.

## Related

- [api-quick-reference.md](api-quick-reference.md): every symbol these terms name.
- [families.md](families.md): which families each structural term applies to.
- [../usage/vocabulary.md](../usage/vocabulary.md), [../usage/residual-stream.md](../usage/residual-stream.md), [../usage/availability.md](../usage/availability.md), [../usage/delta-net.md](../usage/delta-net.md).
- nnsight `docs/reference/glossary.md` for envoy, eproperty, interleaver, mediator, occurrence, source, tracer.
