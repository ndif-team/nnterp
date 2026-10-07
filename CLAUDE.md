# nnterp — Agent Guide

This file routes you to the right page under `docs/` for whatever the user is asking about. The
content lives in `docs/`; **read the matching page before writing code**. The pages are
recipe-style and every snippet in them has been run against the pinned checkpoints in `tests/`.

nnterp is a thin layer on nnsight 0.8: `StandardizedTransformer` is an nnsight `TransformersModel`
whose envoy tree answers to one set of names on every transformer family, with standard values
(`layer_output`, `attention_output`, `attention_probabilities`, ...) that mean the same thing
everywhere. Everything nnsight does (`trace`, `generate`, `.save()`, `tracer.iter`, invokes,
`.source`, remote) works unchanged; nnsight's own guide is `~/wd/nnsight/CLAUDE.md` and its docs
`~/wd/nnsight/docs/`. This file covers only what nnterp adds.

---

## How to use this file

1. Find the user's intent in **"By task"** and follow the link.
2. If the request is about one model family, check **[docs/reference/families.md](docs/reference/families.md)** for its quirks.
3. If a value is missing or raises `Unavailable`, read **[docs/usage/availability.md](docs/usage/availability.md)**.
4. The **inline cheat-sheet** at the bottom lists the mistakes agents make most; internalize it before writing nnterp code.

---

## By task

### "Load a model and use the standard names"
- [docs/usage/loading.md](docs/usage/loading.md) — `StandardizedTransformer(repo_id, ...)`; pass `attn_implementation="eager"` for anything inside attention
- [docs/usage/vocabulary.md](docs/usage/vocabulary.md) — `embed_tokens`, `layers[i].self_attn`, `layers[i].mlp`, `norm`, `lm_head`; native names keep working
- [docs/reference/families.md](docs/reference/families.md) — the 98 families, their native names and quirks

### "Read or edit the residual stream / a sublayer's contribution"
- [docs/usage/residual-stream.md](docs/usage/residual-stream.md) — `layer_output`, `attention_output`, `mlp_output`; `input + attention_output + mlp_output == layer_output`

### "Read or edit attention: the pattern, queries, keys, values, scores, heads"
- [docs/usage/attention-interior.md](docs/usage/attention-interior.md) — `attention_probabilities`, `attention_queries/keys/values/scores/head_outputs`; needs eager; per-family caveats
- [docs/patterns/attention-patterns.md](docs/patterns/attention-patterns.md) — head metrics and pattern edits

### "Mixture of experts: the router, the experts, expert ablation and rerouting"
- [docs/usage/mixture-of-experts.md](docs/usage/mixture-of-experts.md) — `layers[i].mlp` is a `Moe` on the 39 MoE families: `router_logits` (writable, before the scoring), `expert_weights` / `expert_indices` (`[batch, seq, top_k]`), `expert_outputs` (needs `experts_implementation="grouped_mm"`, the default), `routed_output`, `shared_expert_output`; `num_experts`, `top_k`, `SCORING`
- [docs/patterns/expert-ablation.md](docs/patterns/expert-ablation.md) — every expert's effect on a prediction

### "Logits, embeddings, next-token probabilities, the input, the sizes"
- [docs/usage/root-values.md](docs/usage/root-values.md) — `logits`, `token_embeddings`, `next_token_probs`, `input_ids`, `attention_mask`, `input_size`, `num_layers`, `head_dim`, ... (each root size a `StandardizedProperty`: the config's value, by the plain rule or the family's spelling); a block's own sizes on `layers[i].self_attn` (`num_heads`, `num_kv_heads`, `head_dim`, `qk_head_dim`) and `layers[i].mlp` (`intermediate_size`), which differ from the root's on Gemma-4 and MiMo-V2-Flash

### "Vision-language models: images, the vision tower, the projector"
- [docs/usage/vision.md](docs/usage/vision.md) — load with `task="image-text-to-text"`, `model.trace(prompt, images=[image])`; `model.vision.layers[i]` (tower blocks, `Patches`), `model.projector`, the tower's `vision.image_token_mask` and `vision.image_features` (`layers[0].input[vision.image_token_mask] == vision.image_features`, read at the scatter); SigLIP, CLIP, Pixtral (packed, `[1, patches, vision_hidden]`), the Qwen ViT (packed; queries, keys, values and head outputs whole, no pattern; Qwen3-VL's `layers[k].deepstack_output`), the ViTs of Llama 4 (CLS last) and Gemma 4 (padded patches), and Gemma 4 unified's encoder-free embedder (a `vision` with no blocks), across 16 families
- [docs/patterns/image-pathway.md](docs/patterns/image-pathway.md) — ablate the image at `vision.image_features` or inside the tower, patch one image's features into another's run, each head's mass onto the image, one-sided edits at the image positions (`h[mask] = 0`)
- [docs/developing/vision-design.md](docs/developing/vision-design.md) — the design, what does not fit, what is left

### "Does this checkpoint have that value?"
- [docs/usage/availability.md](docs/usage/availability.md) — `model.support()` before the trace; `nnterp.Unavailable` at the read; the reasons you will see

### "Skip layers, steer, logit lens, top-k tokens"
- [docs/usage/methods.md](docs/usage/methods.md) — `skip_layers`, `steer`, `project_on_vocab`, `get_topk_closest_tokens`
- [docs/patterns/logit-lens.md](docs/patterns/logit-lens.md), [docs/patterns/steering.md](docs/patterns/steering.md)

### "Qwen3-Next / Qwen3.5 / OLMo-Hybrid / Kimi-Linear: linear attention, the recurrent state"
- [docs/usage/delta-net.md](docs/usage/delta-net.md) — `linear_attn` values; `route_kernels(model.family, "torch")` for the per-token `state`/`states`
- [docs/patterns/delta-net-state.md](docs/patterns/delta-net-state.md) — patch and track the state

### "Mamba / Falcon-Mamba / Jamba: the selective scan, the state-space state"
- [docs/usage/selective-scan.md](docs/usage/selective-scan.md) — `linear_attn` on a Mamba-1 mixer (`SelectiveScan`): `C`/`B`/`x` as queries/keys/values, `betas` = `dt`, `decays` = `dt * A`; `route_kernels(model.family, "torch")` before the first trace

### "Mamba-2 / Nemotron-H / Bamba / Falcon-H1: the state-space mixer"
- [docs/usage/state-space.md](docs/usage/state-space.md) — `linear_attn` is a `StateSpace`: `C`/`B`/`x` as queries/keys/values, `dt` as `betas`; `route_kernels(model.family, "torch")` when `mamba_ssm` is installed; `nnterp.chunk_per_token(model)` for the state after every token (`states`, `state_after`); `betas`/`decays` assignable

### "What shape is this value?"
- [docs/usage/layouts.md](docs/usage/layouts.md) — one layout per value on every family, named (`Residual`, `Pattern`, `Keys`, ... in `nnterp.components`), except `layer_output` on DeepSeek-V4 (`Streams`) and `linear_attn.decays` on Kimi-Linear (`ChannelGates`); `value.dims`, `value.layout is Pattern`

### "Generation, many prompts, activations datasets"
- [docs/usage/generation.md](docs/usage/generation.md) — the values under `model.generate`, `tracer.iter` picks the step
- [docs/usage/prompt-utils.md](docs/usage/prompt-utils.md) — `nnterp.prompt_utils`: target-token mass over prompts
- [docs/usage/activations.md](docs/usage/activations.md) — `nnterp.nnsight_utils`: `get_token_activations` and friends

### "Run a research pattern across families"
- [docs/patterns/index.md](docs/patterns/index.md) — logit lens, steering, attention patterns, ablation, activation patching, contribution decomposition, cross-family sweep, probing, the image pathway, DeltaNet state

### "Run remotely on NDIF"
- [docs/usage/remote.md](docs/usage/remote.md) — `remote=True`; nnterp installed server-side, never shipped by value

### "Add a family, override a value, add my own value"
- [docs/extending/adding-a-family.md](docs/extending/adding-a-family.md) — one module named after `model_type`, one test file; `def <size>(model)` in the module where the config spells a root size its own way
- [docs/developing/recurrent-mixer-internals.md](docs/developing/recurrent-mixer-internals.md) — a mixer with a recurrent state read at a kernel call (DeltaNet, state-space): subclass `RecurrentMixer`, set `CHUNK_KERNEL` / `RECURRENT_KERNEL` / `STATE_OP`, declare the values
- [docs/extending/overriding-values.md](docs/extending/overriding-values.md) — an `EProperty` keyed on a path (`"../norm.output"`, `"source.<op>.inputs"` with `select`), `unavailable(...)`, `off_interface`, transforms
- [docs/extending/custom-values.md](docs/extending/custom-values.md) — a new `EProperty` (a path from the host: `"output"`, `"../ln_2.output"`, `"source.<op>.output"`) through `envoys=`; annotate `-> Residual` / `-> Pattern` from `nnterp.components`
- [docs/extending/finding-source-ops.md](docs/extending/finding-source-ops.md) — `print(envoy.source)` and how ops are named
- [docs/extending/registering.md](docs/extending/registering.md) — `nnterp.families.register(family, *model_types)`

### "The encyclopedia: a web page per family"
- [encyclopedia/README.md](encyclopedia/README.md) — `encyclopedia/build.py` renders `encyclopedia/entries/<model_type>.py` (hand-written: block schema, quirks, notes; one entry for each of the 98 families) merged with a meta build of the family into static HTML under `encyclopedia/site/`; `entries/gemma2.py` is the reference entry; a family's hue comes from `encyclopedia/hues.py` (add the family to its `LAYOUT` and run it with `--write`, never pick a hue by hand); a page's checkpoint selector swaps every per-checkpoint part (sizes, `support()`, printout) of the entry's `CHECKPOINTS`; an entry's `WRAPPERS` describes its vision-language wrappers (`llama.py` is the reference) and `encyclopedia/vision/` holds each vision encoder two or more families host; `tests/test_encyclopedia.py` builds every entry from its pinned checkpoint and each wrapper from its pinned tiny one

### "Change nnterp itself"
- [docs/developing/index.md](docs/developing/index.md) — architecture, descriptor internals, the recurrent mixer (`RecurrentMixer`, DeltaNet) and its occurrence arithmetic, tests, transformers compatibility, gotchas, contributing
- **Run `HF_HUB_OFFLINE=1 pytest` (about 6800 tests, ~7 min on CPU) before and after.**

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

- **Everything nnsight's cheat-sheet says still holds**: `.save()` and bind the name, reads in forward order within an invoke, nothing assigned in a trace body survives it without a save.
- **Load on one device with `device=`**: `device="cpu"` keeps a model on the CPU; `device_map="cpu"` does not (nnsight's pipeline passes its own `device`, and the model lands on `cuda:0`).
- **Pass `attn_implementation="eager"` at load** if you will touch anything inside attention (`attention_probabilities`, queries, keys, values, scores, head outputs). The default is the checkpoint's, usually `sdpa`, and the values are then unavailable.
- **Check `model.support()` outside the trace, not `hasattr` inside it.** `hasattr(envoy, "attention_probabilities")` never answers `False`: it raises `nnterp.Unavailable` when the value is unavailable, and outside a trace raises nnsight's "Cannot access ... outside of interleaving" for an available one.
- **Target tokens: `ids = model.tokenizer(" Paris", add_special_tokens=False).input_ids` and assert `len(ids) == 1`.** `tokenizer.encode(" Paris")[0]` is BOS on Llama and Gemma (every probability then reads 0.000); Mixtral-8x7B's tokenizer (and the tiny Mistral/Mixtral checkpoints') gives `['▁', '▁Paris']` (try `"Paris"`, the same `▁Paris`), while Mistral-7B v0.1/v0.3 give `▁Paris` whole; on the Tekken vocabularies (Nemo, Ministral-8B, Pixtral-12B) `ĠParis` and `Paris` are different tokens, so keep the space. Granite 3.x's gives `['ĠPar', 'is']` (pick another word; the 4.x tokenizer has ` Paris` whole).
- **Take KLs on log-probabilities** (`model.logits[:, -1].float().log_softmax(-1)`, `F.kl_div(..., log_target=True)`): `next_token_probs` underflows to exact zeros and `p * (p.log() - q.log())` is NaN.
- **Pick blocks from the module lists, not from `num_layers // 2`**: on a hybrid that index is usually a `linear_attn` block with no `self_attn`, and a pure state-space model has none. Decide which blocks have `self_attn` vs `linear_attn` outside the trace; `getattr(envoy, name, None)` inside a trace can trip served values, and `if envoy:` falls through to the module's `__len__`.
- **`layer_output`, `attention_output`, `mlp_output` are tensors on every family**; never index `[0]`. The native `.output` may be a tuple (GPT-J, GPT-Neo, BLOOM, MPT, Falcon).
- **`attention_output` is what the block adds to the stream**, not necessarily the module's return: on Gemma-2/3/4, OLMo-2/3, EXAONE-4, FlexOlmo and OLMo-Hybrid's attention blocks it is the post-norm's output, on BLOOM/MPT/DBRX the pre-residual value. The identity `layers[i].input + attention_output + mlp_output == layer_output` is what you can rely on, except on Gemma-4 (`(... [+ per_layer_output]) * layer_scalar == layer_output`, so a term's weight in the final stream is the product of every later scalar: docs/patterns/contribution-decomposition.md), Doge and ZAYA (per-channel stream gates) and DeepSeek-V4 (parallel streams).
- **Granite, GraniteMoE(-Shared/-Hybrid/-SWA), HyperCLOVA X and ZAYA serve computed copies** (`* residual_multiplier` and the like) and divide the whole edited copy back, so a write at one position moves the others by rounding; in bf16 that can match a small edit's own effect (granite-3.0-1b-a400m, `residual_multiplier` 0.22: a small edit at one position moved earlier positions' logits by 0.17 against 0.19 at the edited one). That rounding is the 0.22 multiplier's (Granite 3.x, granite-4.1-3b/8b, granite-4.0-1b): 0.28, 0.263, 0.246 and 0.175 survive a bf16 divide-and-multiply exactly. Edit the native module output scaled the other way instead (`mlp.output[:, -1] += v / residual_multiplier` leaves the other positions bit-identical), or load in float32. Granite 4.2 sets every multiplier to 1.0, so none of this applies there.
- **`layer_output` is rank 4 on DeepSeek-V4**: `[batch, seq, streams, hidden]`, and `layers[i].input` too; the contributions stay `[batch, seq, hidden]`. Rank-3 code (`resid[:, -1] @ W`, `lm_head(norm(resid))`) runs and answers per stream; `model.project_on_vocab` collapses the streams the way the model does.
- **`model.logits` is the logits the model returns (after any softcap or scale past the head); `lm_head.output` is the raw projection.** `next_token_probs`, `input_size` and `states` are read-only. `token_embeddings` is the embedding module's output, not always what enters block 0 (GPT-2's `wpe`, Granite's `embedding_multiplier` come after it); `layers[0].input` is.
- **Read order traps**: on Falcon without alibi read `attention_values` before `attention_queries`/`attention_keys` (with alibi: queries, then keys, then values); `skip_layers` consumes `layers[start].input`, so read it first; a block's interior values come before its `attention_output`; on a Mamba-1 or Mamba-2 *decode step* read `state_output` before `attention_head_outputs`.
- **An out-of-order read fails loudly only in a plain trace** (`OutOfOrderError`, naming an internal location such as `'...attention_interface_1.fn.i0'`, not the value). Inside `tracer.iter` it binds the value's *next* occurrence, the next step's, so lists come back shifted by one step without an error; only a read whose next occurrence never comes (the last step, a position past the prompt) cuts the block short with a `was never reached` warning that blames the loop. If a saved name is missing or a list looks shifted, check the read order.
- **Two invokes cannot both touch `attention_probabilities` or `attention_scores`** today (`TypeError: 'NoneType' object is not subscriptable`, an nnsight bug); use one trace per prompt or `attention_head_outputs`, which works across invokes. Overwriting or multiplying `attention_scores` lifts the causal mask; add to them, or keep the masked entries.
- **In-place edits on queries, keys and values: assign instead** on GPT-2's and GPT-BigCode's queries (their keys and values take in-place edits), MPT's queries, keys and values, and every recurrent mixer (DeltaNet's q/k/v, Mamba-1's `C`/`B`, Mamba-2's `C`/`B`/`x`), or edit under `torch.no_grad()`. Falcon's `mlp_output` is a copy carried back by a transform; both forms reach the model.
- **Recurrent mixers need `nnterp.route_kernels(model.family, "torch")` before the first trace that reads a value inside the mixer** where an optimized kernel is installed (a plain trace before it, `layer_output` only, does not fix the kernels; the first read inside the mixer does): for DeltaNet's per-token `state`/`states`, and on Mamba-1 (Mamba, Falcon-Mamba, Jamba) and Mamba-2 whenever `mamba_ssm` is installed, whose CUDA kernels have no source and do not run on CPU. On CPU route before any trace: unrouted, even a `layer_output` read fails inside the kernel, with `Expected u.is_cuda()` on Mamba-1 and with `ValueError: Pointer argument cannot be accessed from Triton (cpu tensor?)` (a GPU visible) or `RuntimeError: invalid argument to exchangeDevice` (none) on Mamba-2, Bamba and Falcon-H1. `model.train()` on Mamba-1 calls the fused `mamba_inner_fn`, which `route_kernels` does not reroute: a training-mode trace needs `causal-conv1d` even after routing. DeltaNet's queries and keys are served before the kernel's l2-norm and scale; Mamba-2's `betas` and `decays` are one argument (`dt`): writing `decays` rewrites `betas`, and a write-back of unchanged values is not exact in bf16. A state write is not all a block remembers: a width-4 convolution carries the last tokens.
- **A mixture's routing is the sparse pair `[batch, seq, top_k]`**: ablate expert `e` with `moe.expert_weights = moe.expert_weights.masked_fill(moe.expert_indices == e, 0)`; a rerouted index keeps the old slot's weight. Read a mixture's values in forward order (`router_logits`, weights/indices, `expert_outputs`, `routed_output`); where `shared_expert_output` falls differs per family. Mask pad tokens with `model.attention_mask` when counting usage (the router routes them). Under two or more invokes, edit the routing in place, and take the clean baseline from an unedited invoke of the same batch; sweep single experts in float32. ZAYA's skipped slots read as expert 0 with weight 0 (the tiny skips; on ZAYA1-8B and 74B-preview the balancing biases keep skip from ever being chosen).
- **`envoys=` keys match by module type or native path, never by alias**; to displace a family's envoy, key yours on the type. An `EProperty` path that goes up (`"../ln_2.output"`) takes native names only.
- **Vision-language models: load with `task="image-text-to-text"`** (the default `text-generation` load has no processor, so `model.vision` serves nothing and says so), pass the image as `model.trace(prompt, images=[image])` with the placeholder in the prompt (the processor's chat template puts it there), one image-carrying invoke per trace (several images go in that invoke, as lists). Read `vision.image_token_mask` first (it comes off the inputs), then the tower's values, then `vision.image_features`, then the text model's; on the Qwen ViT the merger (`projector`) runs inside the tower, so read `projector.input`/`.output` before `tower_output`. Boolean indexing with the mask flattens the batch: `out[mask]` is `[image_tokens, hidden]`, `out[~mask]` the text rows; `h[mask] = 0` in a trace is a one-sided edit that lands. Eager traces on a big tower (Gemma 3's 4096 patches) need `torch.no_grad()`; Llama 4 loads in bfloat16 (its processor returns bf16 pixels). Zeroing `image_features` is not the whole image on Qwen3-VL (`layers[0..2].deepstack_output` re-add it) and not reachable through `tower_output` on Llava 1.5 / BakLLaVA (the projector reads block -2) or on llava-interleave, LLaVA-OneVision and Aya Vision (it reads the last block before `vision.norm`).
- **Import nnterp (or nnsight) before any `transformers.models...` module**; the reverse order segfaults on this stack.
- **Every snippet in `docs/` ran against a cached checkpoint**; when a page and the code disagree, the suite is the arbiter: `HF_HUB_OFFLINE=1 pytest tests/families/test_<family>.py`.
