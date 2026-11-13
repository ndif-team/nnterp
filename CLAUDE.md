# nnterp — Agent Guide

This file routes you to the right page under `docs/` for whatever the user is asking about. The
content lives in `docs/`; **read the matching page before writing code**. The pages are
recipe-style and every snippet in them has been run against the pinned checkpoints in `tests/`.

nnterp is a thin layer on nnsight 0.8: `StandardizedTransformer` is an nnsight `TransformersModel`
whose envoy tree answers to one set of names on every transformer family, with standard values
(`layer_output`, `attention_output`, `attention_probabilities`, ...) that mean the same thing
everywhere. Everything nnsight does (`trace`, `generate`, `.save()`, `tracer.iter`, invokes,
`.source`, remote) works unchanged. nnsight is https://github.com/ndif-team/nnsight (its guide is
its `CLAUDE.md`, its docs are under `docs/`), and the `nnsight` plugin of
https://github.com/ndif-team/skills carries its skills. `StandardizedVLLM` is the same over
nnsight's `VLLM` engine. This file covers only what nnterp adds.

This file is for agents working *in this repository*. An agent *using* nnterp from another project
should install the `nnterp` plugin from https://github.com/ndif-team/skills (skills `nnterp`, `patterns`, `extending`)
and read the documentation site, which serves every page as Markdown
(https://ndif-team.github.io/nnterp/llms.txt). When a change here alters what an agent should do,
the matching skill changes too.

---

## How to use this file

1. Find the user's intent in **"By task"** and follow the link.
2. If the request is about one model family, check **[docs/reference/families.md](docs/reference/families.md)** for its quirks. The page is too long to read whole: grep for the family's rows (`` grep -n '`<model_type>`' docs/reference/families.md ``). The [encyclopedia](https://ndif-team.github.io/nnterp/encyclopedia/) has a page per family (`<model_type>.html`): the block drawn with these names and where each value is read.
3. If a value is missing or raises `Unavailable`, read **[docs/usage/availability.md](docs/usage/availability.md)**.
4. The **inline cheat-sheet** at the bottom lists the mistakes agents make most; internalize it before writing nnterp code.

---

## By task

### "Load a model and use the standard names"
- [docs/usage/loading.md](docs/usage/loading.md) — loading a checkpoint and the options: `StandardizedTransformer(repo_id, ...)`, `attn_implementation="eager"`, `device=`, `family=`, `rename=`, `envoys=`
- [docs/usage/vocabulary.md](docs/usage/vocabulary.md) — `embed_tokens`, `layers[i].self_attn`, `layers[i].mlp`, `norm`, `lm_head`; native names keep working
- [docs/reference/families.md](docs/reference/families.md) — the 98 families: native names, relocated values, what `support()` lacks, per-family quirks
- [docs/usage/loading.md#across-gpus-distributed_config](docs/usage/loading.md#across-gpus-distributed_config) — a model split across GPUs: `distributed_config=DistributedConfig(tp_size=N)` under `torchrun`, and the rules for reading values on every rank
- [docs/usage/loading.md#an-architecture-with-no-family](docs/usage/loading.md#an-architecture-with-no-family) — a `model_type` with no family: the best-effort `default` family, what it serves and refuses, `UnsupportedFamily`
- [docs/usage/loading.md#how-the-family-is-chosen](docs/usage/loading.md#how-the-family-is-chosen) — which family a checkpoint gets, before loading: `nnterp.families.covers(model_type)`, `resolve(config)`, `known()`, `model.family`

### "Read or edit the residual stream / a sublayer's contribution"
- [docs/usage/residual-stream.md](docs/usage/residual-stream.md) — the stream and what each sublayer adds: `layer_input`, `layer_output`, `attention_output`, `mlp_output`, and the boundaries between them

### "Read or edit attention: the pattern, queries, keys, values, scores, heads"
- [docs/usage/attention-interior.md](docs/usage/attention-interior.md) — `attention_probabilities`, `attention_queries/keys/values/scores/head_outputs`; needs eager; per-family caveats
- [docs/usage/attention-interior.md#before-the-rotary-embedding-attention_queries_prerope-attention_keys_prerope-rotate](docs/usage/attention-interior.md#before-the-rotary-embedding-attention_queries_prerope-attention_keys_prerope-rotate) — queries and keys before RoPE, to move or edit them before the rotation: `attention_queries_prerope`, `attention_keys_prerope`, `self_attn.rotate(x, positions)`
- [docs/patterns/attention-patterns.md](docs/patterns/attention-patterns.md) — head metrics and pattern edits
- [docs/usage/attention-interior.md#calling-the-attention-function-again-interface](docs/usage/attention-interior.md#calling-the-attention-function-again-interface) — rerunning the block's attention on edited queries, keys or values: `self_attn.interface(q, k, v)` -> `(head_outputs, pattern)`, and its read order
- [docs/patterns/head-kv-patching.md](docs/patterns/head-kv-patching.md) — chosen heads read another run's keys and values: clone, replace, `interface`, write `attention_head_outputs`

### "Mixture of experts: the router, the experts, expert ablation and rerouting"
- [docs/usage/mixture-of-experts.md](docs/usage/mixture-of-experts.md) — reading, ablating, rerouting experts on `layers[i].mlp`: `router_logits`, `expert_weights`, `expert_indices`, `expert_outputs`, `routed_output`, ...
- [docs/patterns/expert-ablation.md](docs/patterns/expert-ablation.md) — every expert's effect on a prediction

### "Logits, embeddings, next-token probabilities, the input, the sizes"
- [docs/usage/root-values.md](docs/usage/root-values.md) — whole-model values and sizes: `logits`, `token_embeddings`, `next_token_probs`, `input_ids`, `num_layers`, `head_dim`, ...; a block's own sizes

### "Vision-language models: images, the vision tower, the projector"
- [docs/usage/vision.md](docs/usage/vision.md) — `task="image-text-to-text"`, `images=[...]`, the tower (`model.vision.layers[i]`), `model.projector`, `vision.image_token_mask`, `vision.image_features`
- [docs/patterns/image-pathway.md](docs/patterns/image-pathway.md) — ablating or patching what the image contributes: at `vision.image_features`, inside the tower, each head's mass on the image
- [docs/developing/vision-design.md](docs/developing/vision-design.md) — the design, what does not fit

### "Does this checkpoint have that value?"
- [docs/usage/availability.md](docs/usage/availability.md) — `model.support()` before the trace; `nnterp.Unavailable` at the read; the reasons you will see
- [docs/usage/availability.md#sourcenotavailable-the-forward-took-another-path](docs/usage/availability.md#sourcenotavailable-the-forward-took-another-path) — a read raised `SourceNotAvailable`: this run's forward took a path without the value's operation, which `support()` cannot predict

### "Several reads or edits in one trace: put them in forward order"
- [docs/usage/availability.md#forward-order-order-and-rank](docs/usage/availability.md#forward-order-order-and-rank) — ordering reads and edits on any family: sort by `model.rank(name, layer)`, list a block's with `model.order(i)`; explains an `OutOfOrderError`
- [docs/usage/generation.md#read-order-within-a-step](docs/usage/generation.md#read-order-within-a-step) — why a misplaced read under `tracer.iter` can return the next step's value instead of raising

### "Skip layers, steer, logit lens, top-k tokens"
- [docs/usage/methods.md](docs/usage/methods.md) — `skip_layers`, `steer`, `project_on_vocab`, `get_topk_closest_tokens`, `probs_to_dict`
- [docs/patterns/logit-lens.md](docs/patterns/logit-lens.md), [docs/patterns/steering.md](docs/patterns/steering.md)

### "One attention head or one MLP neuron: ablate, patch, attribute"
- [docs/usage/attention-interior.md#attention_premix-the-heads-at-the-output-projection](docs/usage/attention-interior.md#attention_premix-the-heads-at-the-output-projection) — one head's output under any `attn_implementation`, for per-head ablation and attribution: `self_attn.attention_premix`
- [docs/usage/residual-stream.md#the-module-boundaries](docs/usage/residual-stream.md#the-module-boundaries) — one MLP neuron: `mlp.mlp_neurons` (what `down_proj` sums) vs `mlp.mlp_activation` (`act_fn`'s output)
- [docs/patterns/ablation.md](docs/patterns/ablation.md), [docs/patterns/contribution-decomposition.md](docs/patterns/contribution-decomposition.md) — head ablation; per-head direct logit attribution
- [docs/usage/attention-interior.md#the-weights-per-head-value_weight-output_weight-attention_input_gain](docs/usage/attention-interior.md#the-weights-per-head-value_weight-output_weight-attention_input_gain) — a head's weights for OV circuits and attribution: `self_attn.value_weight(kv)`, `output_weight(j)`, `layers[i].attention_input_gain` / `attention_output_gain`

### "Qwen3-Next / Qwen3.5 / OLMo-Hybrid / Kimi-Linear: linear attention, the recurrent state"
- [docs/usage/delta-net.md](docs/usage/delta-net.md) — `linear_attn` values; `route_kernels(model.family, "torch")` for the per-token `state`/`states`
- [docs/patterns/delta-net-state.md](docs/patterns/delta-net-state.md) — patch and track the state

### "Mamba / Falcon-Mamba / Jamba: the selective scan, the state-space state"
- [docs/usage/selective-scan.md](docs/usage/selective-scan.md) — a Mamba-1 mixer as `linear_attn` (`SelectiveScan`): `C`/`B`/`x` as queries/keys/values, `betas`, `decays`, the state; `route_kernels` first

### "Mamba-2 / Nemotron-H / Bamba / Falcon-H1 / GraniteMoE-Hybrid: the state-space mixer"
- [docs/usage/state-space.md](docs/usage/state-space.md) — a Mamba-2 mixer as `linear_attn` (`StateSpace`): `C`/`B`/`x`, `betas`/`decays`, the state; `nnterp.chunk_per_token(model)` for the state after every token

### "What shape is this value?"
- [docs/usage/layouts.md](docs/usage/layouts.md) — every value's named layout (`Residual`, `Pattern`, `Keys`, ...) and its exceptions: `value.layout`, `value.dims`, `nnterp.axis_size(envoy, axis)`

### "Generation, many prompts, activations datasets"
- [docs/usage/generation.md](docs/usage/generation.md) — the values under `model.generate`, `tracer.iter` picks the step
- [docs/usage/prompt-utils.md](docs/usage/prompt-utils.md) — `nnterp.prompt_utils`: target-token mass over prompts
- [docs/usage/activations.md](docs/usage/activations.md) — `nnterp.nnsight_utils`: `get_token_activations` and friends

### "Run a research pattern across families"
- [docs/patterns/index.md](docs/patterns/index.md) — recipes that run on every family: logit lens, steering, attention patterns, ablation, activation patching, probing, ...

### "Run on the vLLM engine"
- [docs/usage/vllm.md](docs/usage/vllm.md) — `StandardizedVLLM(repo_id, dispatch=True, ...)`: the same names and values on nnsight's `VLLM`, what differs, adding a vLLM family

### "Run remotely on NDIF"
- [docs/usage/remote.md](docs/usage/remote.md) — `remote=True`; nnterp installed server-side, never shipped by value

### "Add a family, override a value, add my own value"
- [docs/extending/adding-a-family.md](docs/extending/adding-a-family.md) — a new family: one module named after `model_type`, one test file; `def <size>(model)` where the config spells a root size its own way
- [docs/developing/recurrent-mixer-internals.md](docs/developing/recurrent-mixer-internals.md) — a mixer with a recurrent state read at a kernel call: subclass `RecurrentMixer`, set `CHUNK_KERNEL` / `RECURRENT_KERNEL` / `STATE_OP`
- [docs/extending/overriding-values.md](docs/extending/overriding-values.md) — pointing a value elsewhere: an `EProperty` keyed on a path, `unavailable(...)`, `off_interface`, transforms
- [docs/extending/custom-values.md](docs/extending/custom-values.md) — a value of your own: a new `EProperty` through `envoys=`, annotated with a layout (`-> Residual`)
- [docs/extending/finding-source-ops.md](docs/extending/finding-source-ops.md) — `print(envoy.source)` and how ops are named
- [docs/extending/registering.md](docs/extending/registering.md) — a family from outside nnterp: `nnterp.families.register(family, *model_types)`; for one load, `StandardizedTransformer(repo_id, family=...)`

### "The encyclopedia: a web page per family"
- [encyclopedia/README.md](encyclopedia/README.md) — a family's page: `entries/<model_type>.py` (reference: `entries/gemma2.py`; for `WRAPPERS`, `entries/llama.py`), `build.py`, `hues.py --write`, `tests/test_encyclopedia.py`

### "Change nnterp itself"
- [docs/developing/index.md](docs/developing/index.md) — internals: architecture, descriptors, `RecurrentMixer` and its occurrence arithmetic, tests, transformers compatibility, gotchas, contributing, releasing
- **Run `HF_HUB_OFFLINE=1 pytest -n 8` before and after.**

### "Every symbol / every term"
- [docs/reference/api-quick-reference.md](docs/reference/api-quick-reference.md), [docs/reference/glossary.md](docs/reference/glossary.md)

---

## Folders

| Folder | What it holds | Start at |
|---|---|---|
| `docs/usage/` | one page per feature: loading, names, every standard value, methods, hybrids, helpers, remote | [docs/usage/index.md](docs/usage/index.md) |
| `docs/patterns/` | interpretability recipes written once against the standard values, so they run on every family | [docs/patterns/index.md](docs/patterns/index.md) |
| `docs/extending/` | adding a family, overriding a value, adding your own values, registering from outside nnterp | [docs/extending/index.md](docs/extending/index.md) |
| `docs/developing/` | internals: architecture, the descriptors, the recurrent mixer and its occurrence arithmetic, tests, compatibility, gotchas | [docs/developing/index.md](docs/developing/index.md) |
| `docs/reference/` | API quick reference, the families table, glossary | [docs/reference/api-quick-reference.md](docs/reference/api-quick-reference.md) |

---

## Inline cheat-sheet (read before writing nnterp code)

- **Everything nnsight's cheat-sheet says still holds**: `.save()` and bind the name, reads in forward order within an invoke, nothing assigned in a trace body survives it without a save. That includes containers: a list first bound inside the body is gone after it even when its elements are saved; bind it before the trace (`xs = []`) or save the list (`nnsight.save([...])`).
- **Load on one device with `device=`**: `device="cpu"` keeps a model on the CPU; `device_map="cpu"` does not (the pipeline the model loads through moves a model that sits on one device to the first GPU).
- **Pass `attn_implementation="eager"` at load** if you will touch anything inside attention (`attention_probabilities`, queries, keys, values, scores, head outputs). The default is the checkpoint's, usually `sdpa`, and the values are then unavailable.
- **Check `model.support()` outside the trace, not `hasattr` inside it.** `hasattr(envoy, "attention_probabilities")` never answers `False`: it raises `nnterp.Unavailable` when the value is unavailable, and outside a trace raises for an available one too (nnsight's `ValueError: Cannot access ... outside of interleaving`, or `SourceNotAvailable` for a value inside a nested call, such as the pattern).
- **Target tokens: `ids = model.tokenizer(" Paris", add_special_tokens=False).input_ids` and assert `len(ids) == 1`.** `tokenizer.encode(" Paris")[0]` is the BOS token wherever the tokenizer adds one (Llama, Gemma, Mistral; every probability then reads 0.000). Tokenizers differ per checkpoint: some split a leading-space word in two, dropping the space gives the same token on some and a different one on others, and some append a token to every prompt, so the last position is not the prediction after your text. Check the ids on the checkpoint you use; the known cases are under the family in [docs/reference/families.md](docs/reference/families.md).
- **Take KLs on log-probabilities** (`model.logits[:, -1].float().log_softmax(-1)`, `F.kl_div(..., log_target=True)`): `next_token_probs` underflows to exact zeros and `p * (p.log() - q.log())` is NaN.
- **Pick blocks from the module lists, not from `num_layers // 2`**: on a hybrid that index is usually a `linear_attn` block with no `self_attn`, and a pure state-space model has none. Decide which blocks have `self_attn` vs `linear_attn` outside the trace; `getattr(envoy, name, None)` inside a trace can trip served values.
- **`layer_output`, `attention_output`, `mlp_output` are tensors on every family**; never index `[0]`. The native `.output` may be a tuple ([tuple blocks](docs/reference/families.md#tuple-blocks)).
- **`attention_output` is what the block adds to the stream**, not necessarily the module's return: a post-norm's output on a sandwich block, the value before the add where the module adds the residual itself. Rely on `layers[i].input + attention_output + mlp_output == layer_output` only on a block [docs/reference/families.md](docs/reference/families.md) gives no other formula for (a scaled or gated stream, parallel streams, a missing or extra sublayer; Gemma-4, Doge, ZAYA and DeepSeek-V4 among them).
- **On a family whose contributions are computed copies (`* residual_multiplier` and the like: the Granite families, HyperCLOVA X, ZAYA), a write divides the whole edited copy back**, so the positions you did not edit move by rounding, in bf16 by as much as a small edit's own effect. Edit the native module output scaled the other way, or load in float32 ([families.md](docs/reference/families.md#scaled-residual-adds)).
- **`layer_output` is not rank 3 on every family**: on a hyper-connection residual (DeepSeek-V4) it is `[batch, seq, streams, hidden_size]`, and `layers[i].input` too, while the contributions and what each sublayer is called with (`attention_norm_output`, `mlp_norm_output`, `mlp_input`) stay `[batch, seq, hidden_size]`. Rank-3 code (`resid[:, -1] @ W`, `lm_head(norm(resid))`) runs and answers per stream; check the value's `layout` in cross-family code, and use `model.project_on_vocab`, which collapses the streams the way the model does.
- **`model.logits` is the logits the model returns (after any softcap or scale past the head); `lm_head.output` is the raw projection.** `next_token_probs`, `input_size` and `states` are read-only. `token_embeddings` is the embedding module's output, not always what enters block 0 (GPT-2's `wpe`, Granite's `embedding_multiplier` come after it); `layers[0].input` is.
- **Read order traps**: a block's interior values come before its `attention_output`; `skip_layers` consumes `layers[start].input`, so read it first; a recurrent mixer's submodule value (`in_proj_qkv.output`, `x_proj.output`, `in_proj.output`) and its kernel values (`attention_queries`, `betas`, ...) cannot be read in one trace in either order: use two traces; on a Mamba-1 or Mamba-2 *decode step* read `state_output` before `attention_head_outputs`. Some families order the attention's own values differently ([families.md](docs/reference/families.md#in-place-edits-and-read-order)): to put reads or edits across values and blocks into forward order, sort them by `model.rank(name, layer)`; `model.order(layer)` lists a block's values in order ([docs/usage/availability.md](docs/usage/availability.md#forward-order-order-and-rank)).
- **An out-of-order read raises in a plain trace, and under `tracer.iter` it can bind the next step instead** (`OutOfOrderError`, naming an internal location such as `'...attention_interface_1.fn.i0'`, not the value). A `tracer.iter` loop that starts past step 0 (`iter[1:3]`) binds every late read to the value's *next* occurrence, the next step's, so lists come back shifted by one step without an error. A loop from step 0 (`iter[:N]`) raises as a plain trace does, except for a value inside a call the forward makes (the scores, the pattern) and one at an operation no earlier read on this model has reached (no earlier trace read it), which bind the next step too. Only a read whose next occurrence never comes (the last step, a position past the prompt) cuts the block short with a `was never reached` warning that blames the loop. If a saved name is missing or a list looks shifted, check the read order. One `OutOfOrderError` is not about order: on a fresh model, reading a child module's value (`self_attn.q_proj.output`) and then an interior value of that module (`attention_queries`) in one trace fails on the first such trace only, because nnsight builds the module's `.source` lazily while its forward is already running; once the model is dispatched (`dispatch=True`, or after its first trace), touch `model.layers[i].self_attn.source` outside the trace, or split the reads into two traces.
- **`attn.interface`: read it once per forward, where the queries, keys and values are** (in any order with them), and make its first call before reading the scores, the pattern or the head outputs; call the result as often as you need, later calls after those reads included. A second read, or a first call after those reads, raises `OutOfOrderError` in a plain trace and in a `tracer.iter` loop from step 0, and lands on the next step's call in a loop that starts later (a shape-mismatch `RuntimeError` at the call, or a loop cut short with a `was never reached` warning). Within one invoke the result is the served head outputs and pattern bit for bit; across invokes it agrees to rounding (the model ran every invoke's rows in one call), so put the prompts in one invoke (`model.trace([a, b])`) when bit for bit matters.
- **A head's weights are `self_attn.value_weight(kv)` / `output_weight(j)` and `layers[i].attention_input_gain`, not slices of `v_proj.weight` / `o_proj.weight`**: under a PEFT adapter `.weight` is the base weight alone, while the accessors give the weight the LoRA layer applies (the base plus every active, unmerged adapter's delta, what `merge()` would leave); several families fuse or transpose the projections ([per-head weights](docs/reference/families.md#per-head-weights)); the pre-attention norm's gain is not always its `weight`; and a post-norm block or a post-attention norm norms `output_weight`'s image before the stream, so fold `attention_output_gain` too. The gains are float32; the weights are the parameter's dtype, live views. They read the module's weights, so the model must be dispatched (`dispatch=True`, or after its first trace): on an undispatched or quantized model the two weights raise `Unavailable`.
- **Two invokes of one trace can both read and edit `attention_probabilities` and `attention_scores`**; at one location the invoke written first is served first, so an empty `tracer.invoke()` after an invoke that edits the scores reads the edited pattern. Overwriting or multiplying `attention_scores` lifts the causal mask; add to them, or keep the masked entries.
- **Assigning `layer_mid` is an in-place write into the stream** (the block holds it as the MLP norm's input and as `residual`, and both must see the write): `layers[i].layer_mid = new` replaces the stream for the MLP and the add after it, the same write as `layer_mid[:] = new`. On transformers that changes a read of `layer_mid` saved earlier in the trace, and a replacement computed from such a read by an op that keeps its input for backward (a learned scale or rotation, a trainable SAE) fails in the backward pass: start from `layers[i].layer_mid.clone()`. A sum (`layer_mid + v`) needs no clone; on vLLM a read is a private copy anyway.
- **In-place edits on queries, keys and values: assign instead where torch refuses them**, or edit under `torch.no_grad()`: on every recurrent mixer (DeltaNet's q/k/v, Mamba-1's `C`/`B`, Mamba-2's `C`/`B`/`x`) and on some families' attention ([in-place edits](docs/reference/families.md#in-place-edits-and-read-order)).
- **Recurrent mixers need `nnterp.route_kernels(model.family, "torch")` before the first trace that reads a value inside the mixer**: for DeltaNet's per-token `state`/`states`, on Mamba-1 for the scan's head outputs and state on every machine, and on Mamba-1 and Mamba-2 for every kernel value when `mamba_ssm` is installed. A plain trace before it (`layer_output` only) does not fix the kernels; the first read inside the mixer does. On CPU with `mamba_ssm` installed, route before any trace: unrouted, even a `layer_output` read fails inside the Mamba kernels. A state write is not all a block remembers: a short convolution carries the last tokens. Per mixer: [delta-net](docs/usage/delta-net.md), [selective-scan](docs/usage/selective-scan.md), [state-space](docs/usage/state-space.md).
- **A mixture's routing is the sparse pair `[batch, seq, top_k]`**: ablate expert `e` with `idx = moe.expert_indices` then `moe.expert_weights = moe.expert_weights.masked_fill(idx == e, 0)` (indices first: JetMoE computes them before the weights); a rerouted index keeps the old slot's weight. Read a mixture's values in forward order (`router_logits`, weights/indices, `expert_outputs`, `routed_output`); where `shared_expert_output` falls differs per family. Mask pad tokens with `model.attention_mask` when counting usage (the router routes them). Under two or more invokes, take the clean baseline from an unedited invoke of the same batch; sweep single experts in float32.
- **On `StandardizedVLLM` use `layer_input` / `layer_output`, never `layers[i].input` / `.output`**: natively the input is the positions and the output a `(hidden_states, residual)` pair on most families. Values there are `[1, tokens, ...]` private copies, `logits` is `[1, 1, vocab_size]`, sampling settings go on `trace` (vLLM's defaults otherwise), the pattern and scores are recomputed from the queries and keys (read-only, prefill only; `Unavailable` on a decode step), a gated MLP's `mlp_activation` is unavailable (vLLM fuses it with the multiply by the up projection), `lm_head_output` is `logits` itself except where vLLM's logits processor softcaps or scales (unavailable there, naming which), `self_attn.rotate` runs inside a trace only (it calls the engine's rotary embedding), and an assigned value must keep the shape it was served with. `model.rank(name, layer)` and `model.order(layer)` work there (outside a trace, the transformers model's ranks for the values the engine serves), and `model.generation_config` is the checkpoint's. The suite is `tests/vllm_families/` and needs vLLM and a GPU.
- **`envoys=` keys match by module type, then path** (native, or any alias spelling of the path: `"self_attn"` reaches GPT-2's `attn`, `"layers.*"` every block); to displace a family's envoy, key yours on the type. An `EProperty` path that goes up (`"../ln_2.output"`) takes native names only.
- **Vision-language models: load with `task="image-text-to-text"`** (the default `text-generation` load has no processor, so `model.vision` serves nothing and says so), pass the image as `model.trace(prompt, images=[image])` with the placeholder in the prompt (the processor's chat template puts it there), one image-carrying invoke per trace (several images go in that invoke, as lists). Read `vision.image_token_mask` first (it comes off the inputs), then `token_embeddings` (the wrapper embeds the ids before it runs the tower), then the tower's values, then `vision.image_features`, then the text model's; on the Qwen ViT the merger (`projector`) runs inside the tower, so read `projector.input`/`.output` before `tower_output`. Boolean indexing with the mask flattens the batch: `out[mask]` is `[image_tokens, hidden_size]`, `out[~mask]` the text rows; `h[mask] = 0` in a trace is a one-sided edit that lands. An eager trace over a large tower needs `torch.no_grad()`, and a processor that returns bfloat16 pixels needs a bfloat16 load (a float32 tower refuses them). A tower edit reaches the text model only through what the projector reads, which is not always `tower_output`, and on some wrappers the image enters the text model at more than one place: check the wrapper in [docs/usage/vision.md](docs/usage/vision.md).
- **Every snippet in `docs/` ran against a cached checkpoint**; when a page and the code disagree, the suite is the arbiter: `HF_HUB_OFFLINE=1 pytest tests/families/test_<family>.py`.
