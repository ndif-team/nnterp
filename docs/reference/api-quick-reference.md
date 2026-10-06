---
title: API Quick Reference
one_liner: Every public nnterp symbol in one place: the model class, the standard values by host with their layouts and availability, the descriptors, the registry, the helper modules and the exceptions.
tags: [reference, api, values, layouts, availability]
related: [docs/usage/loading.md, docs/usage/vocabulary.md, docs/usage/root-values.md, docs/usage/methods.md, docs/usage/availability.md, docs/usage/layouts.md, docs/usage/attention-interior.md, docs/usage/delta-net.md, docs/usage/prompt-utils.md, docs/usage/activations.md, docs/extending/custom-values.md, docs/extending/registering.md, docs/developing/eproperty-internals.md, docs/reference/families.md, docs/reference/glossary.md]
sources: [nnterp/__init__.py, nnterp/standardized.py, nnterp/components/__init__.py, nnterp/components/eproperty.py, nnterp/components/standard.py, nnterp/components/layer.py, nnterp/components/attention.py, nnterp/components/mlp.py, nnterp/components/linear_attention.py, nnterp/components/recurrent.py, nnterp/families/__init__.py, nnterp/prompt_utils.py, nnterp/nnsight_utils.py]
---

# API Quick Reference

## What this is for

One page with every name nnterp exports, its signature as the source declares it, and for every standard value its layout (by name; the axes are in the [layouts table](#layouts)), whether it can be assigned, and what decides its availability. `model` is a `StandardizedTransformer`. Everything nnsight's `TransformersModel` offers (`trace`, `generate`, `session`, `edit`, `tracer.iter`, `.save()`, `remote=`) is inherited unchanged and is documented in nnsight `docs/reference/api-quick-reference.md`; this page covers only what nnterp adds. Signatures are taken from the source; the value rows are what `Standard.values()` reports on each host class.

## Canonical pattern

```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", dispatch=True, attn_implementation="eager")

print(model.family.__name__)                  # nnterp.families.gpt2
print(model.support())                         # {value: None | {layer: reason}}; nothing runs

with model.trace("The Eiffel Tower is in"):
    x = model.layers[3].input.save()
    pattern = model.layers[3].self_attn.attention_probabilities.save()   # inside the attention: before its output
    attn = model.layers[3].self_attn.attention_output.save()
    mlp = model.layers[3].mlp.mlp_output.save()
    resid = model.layers[3].layer_output.save()
    logits = model.logits.save()

torch.testing.assert_close(x + attn + mlp, resid)    # the contribution identity, on every family
print(pattern.shape)                                 # [batch, heads, query, key]
```

Reads within one trace follow the forward: the pattern is produced inside block 3's attention, so it is read before `attention_output`, which is read before `mlp_output`, which is read before `layer_output`.

## Top-level names

`from nnterp import ...` exports exactly these (`nnterp.__all__`):

| Name | What it is |
|---|---|
| `StandardizedTransformer` | The model class: a `TransformersModel` renamed to the standard vocabulary and wrapped in the family's envoys. |
| `Layer`, `Attention`, `Mlp`, `LinearAttention`, `StateSpace` | The base envoys a family subclasses; the hosts of the standard values. |
| `RecurrentMixer` | The base of `LinearAttention` and `StateSpace`: how a recurrent mixer's values are reached at its kernel call, the per-token state and the kernel routing. |
| `Standard` | The envoy base of them all, with `values()`, `support()` and the `sourced` flag. |
| `EProperty`, `DerivedEProperty` | The descriptors a value is made of: one keyed on a path from the host, one computed. |
| `unavailable`, `route_kernels`, `route_delta_rule` | A value a family lacks; the recurrent kernel switch, and its DeltaNet spelling. |
| `chunk_per_token` | A Mamba-2 model's chunk scan with a chunk size of 1, so `StateSpace.states` reads the state after every token. |
| `Unavailable`, `UnsupportedFamily` | The two exceptions nnterp raises itself. |
| `Vision` | A vision tower's root envoy (`model.vision`): its blocks at `layers` (`VisionLayer`, with `VisionAttention`, `VisionMlp` from `nnterp.components`), its sizes, `image_token_mask`, `patch_embeddings`, `tower_output`, `image_features`, `support()`; `PixtralVision` for Pixtral's packed tower, `QwenVision` and `QwenVisionAttention` for the Qwen ViT, and `ImageScatter`, keyed on the wrapper's model, for where `image_features` is read. See [../usage/vision.md](../usage/vision.md). |

`nnterp.families`, `nnterp.components`, `nnterp.prompt_utils` and `nnterp.nnsight_utils` are imported as modules.

## `StandardizedTransformer`

### Constructor

```python
StandardizedTransformer(repo_id, *args, rename=None, envoys=None, tokenizer_kwargs=None, **kwargs)
```

| Argument | Type | Meaning |
|---|---|---|
| `repo_id` | `str` or `torch.nn.Module` | A Hub repo id, or an already-loaded module (its own `config` is read). |
| `rename` | `dict[str, str \| list[str]] \| None` | Extra nnsight aliases, merged over the family's `RENAME`; a key given here wins. |
| `envoys` | `dict \| None` | Extra `envoys=` entries, merged over the family's `ENVOYS` (and nnsight's tensor-parallel envoys on a sharded load); a key given here wins. Keys are module types or native paths, never aliases. |
| `tokenizer_kwargs` | `dict \| None` | Attributes set on the loaded tokenizer: `{"padding_side": "left"}`, a `pad_token`. |
| `**kwargs` | | Passed to `TransformersModel`: `dispatch=True`, `attn_implementation="eager"`, `dtype=`, `device=` (one device; `device_map="cpu"` does not keep a model off the GPU), `device_map=`, `revision=`, `trust_remote_code=`. `task` defaults to `"text-generation"`; `task="image-text-to-text"` loads a multimodal checkpoint as its wrapper, with its processor. |

The constructor reads the checkpoint's config first (`AutoConfig`; a multimodal config's `text_config`), looks up `config.model_type` in `nnterp.families`, and raises `UnsupportedFamily` before any weights load when no family covers it. `attn_implementation` is not forced: the checkpoint's own default (`sdpa` on most) stays, and the interior attention values then report unavailable in `support()`.

### Root values, inside a trace

Every row is an `EProperty` on the root, listed in `repr(model)` with its description and reported by `support()`.

| Value | Layout | Assignable | Description (as the repr shows it) |
|---|---|---|---|
| `model.logits` | `Logits` | yes: replaces `output.logits` | The logits the model returns, after any softcap (Gemma-2) or scale (Granite, Cohere) past the head; `model.lm_head.output` is the raw projection. |
| `model.token_embeddings` | `Residual` | yes | The embedding module's output, `embed_tokens.output`; `layers[0].input` is what enters block 0, after any positional embeddings and embedding norms. An `EProperty` keyed `"embed_tokens.output"`. |
| `model.next_token_probs` | `NextTokenProbs` | no (`AttributeError`: assign `logits`) | `logits[:, -1].softmax(-1)`; the last position is every row's last token only under left padding. |
| `model.input_ids` | `Tokens` | yes: the model runs on the ids you set | The token ids the model was called with. |
| `model.attention_mask` | `Tokens` | yes | The attention mask the model was called with; zeros are padding. |
| `model.input_size` | none (a `torch.Size`) | no (`AttributeError`: assign `input_ids`) | `[batch, seq]` of the current call. |

`input_ids`, `attention_mask` and `input_size` are served at the model's input, the first location of a run: read them before any block's value in the same trace.

### Methods

| Method | Signature | Where | What |
|---|---|---|---|
| `skip_layers` | `skip_layers(start: int, end: int, skip_with: Tensor \| None = None) -> None` | inside a trace, before block `start` runs | Blocks `start..end` inclusive do not run; block `start`'s input (or `skip_with`) becomes each one's `layer_output`, packed as the family's block returns it (`Layer.skip_with`). Negative indices count from the end. |
| `steer` | `steer(layers: int \| list[int], vector: Tensor, factor: float = 1.0, token_positions: int \| list[int] \| slice \| None = None, batch_index: int \| None = None) -> None` | inside a trace, `layers` ascending | Adds `factor * vector` in place to `layer_output` of each block, at the given positions and row (default all). |
| `project_on_vocab` | `project_on_vocab(hidden: Tensor) -> Tensor` | inside (on a live value) or outside (on a saved one) | The logit lens: `lm_head(norm(hidden))`, then what the model does to the head's output to make its logits: the text config's `final_logit_softcapping` if set (Gemma-2), else nothing; a family's `def project_on_vocab(model, hidden)` is bound in its place (Cohere's `* logit_scale`, Granite's `/ logits_scaling`). On the last block's `layer_output` it equals `logits`. |
| `get_topk_closest_tokens` | `get_topk_closest_tokens(hidden: Tensor, k: int = 5) -> list[dict[str, float]]` | outside, on a saved `[..., hidden]` tensor | `project_on_vocab` then softmax; one `{token: probability}` per position, row-major over the leading axes. Takes a residual-stream tensor, not logits. |
| `probs_to_dict` | `probs_to_dict(probs: Tensor, k: int = 5) -> dict[str, float]` | outside | The `k` most likely tokens of one `[vocab]` distribution. |
| `support` | `support(layer: int \| None = None) -> dict[str, Any]` | outside; nothing runs | Without `layer`: every root value and every block value, `None` when available on every block, else `{layer: reason}`. With `layer`: that block's values by dotted name (`"self_attn.attention_probabilities"`), `None` or the reason, including `"no <module> module on this block"`. The keys come from the tree: every `Standard` child of any block, under its standard name, so a value added through `envoys=` is listed as `self_attn.<name>`, a module some blocks lack (a hybrid's `self_attn`) is reported missing on those, and a module no block has (OPT's `mlp`) has no key. A `Standard` child of the root (the vision tower) adds its own `support()` rows under its standard name (`"vision.image_features"`); a tower no image reaches adds none. |

### Sizes, outside a trace

Each is a `StandardizedProperty`: it reads the config by the plain rule unless the model's family module defines a function of the same name (`def intermediate_size(model)` in `gpt2.py`), which then answers. What each family reads is in [../usage/root-values.md](../usage/root-values.md#sizes). A root size is the config's value, equal to every block's where the blocks agree; on Gemma-4 and MiMo-V2-Flash, whose blocks differ, each block's own is on its [`Attention`](#attention) and [`Mlp`](#mlp).

| Property | Plain rule | Family spellings |
|---|---|---|
| `num_layers` | `len(model.layers)` | |
| `hidden_size` | `config.hidden_size` | |
| `vocab_size` | `config.vocab_size` | |
| `num_heads` | `config.num_attention_heads` | |
| `num_kv_heads` | `config.num_key_value_heads`, else `num_heads`. | Falcon: `config.num_kv_heads` under `new_decoder_architecture`, `1` under `multi_query`, else `num_heads`. |
| `head_dim` | `config.head_dim` when the config says (Qwen3, Gemma), else `hidden_size // num_heads`. | DeepSeek-V2/V3: `config.v_head_dim`. |
| `qk_head_dim` | `head_dim`. | DeepSeek-V2/V3: `qk_nope_head_dim + qk_rope_head_dim`. |
| `intermediate_size` | `config.intermediate_size`. A mixture of experts' experts are `config.moe_intermediate_size` wide instead. | GPT-2, GPT-J, CodeGen: `config.n_inner`, `4 * hidden_size` when `None`; GPT-Neo: `config.intermediate_size`, `4 * hidden_size` when `None`; Falcon: `config.ffn_hidden_size`; OPT, XGLM: `config.ffn_dim`; GPT-NeoX-Japanese: `hidden_size * intermediate_multiple_size`; MPT: `expansion_ratio * hidden_size`; BLOOM: `4 * hidden_size`. |

| Name | Signature | What |
|---|---|---|
| `StandardizedProperty` | `nnterp.standardized.StandardizedProperty(fget)` | The descriptor each size is. `__get__` calls `getattr(model.family, <name>)(model)` when the family defines it, else `fget(model)`; on the class it returns itself (`StandardizedTransformer.head_dim`). `__set__` raises `AttributeError("<name> is read off the config; a family defines `def <name>(model)` to say it otherwise")`. Not an `EProperty`: no location, nothing served inside a trace, no entry in the repr or `support()`. |
| `StandardizedCapability` | `nnterp.standardized.StandardizedCapability(fget)` | `StandardizedProperty` for a method, and its subclass: on attribute access it binds `getattr(model.family, <name>)` to the model when the family defines it, else `fget`; on the class it returns itself (`StandardizedTransformer.project_on_vocab`). `project_on_vocab` is the one today. |

### Other attributes

| Attribute | What it is |
|---|---|
| `model.family` | The family module the checkpoint resolved to (`nnterp.families.llama`); what `route_kernels` takes. |
| `model.layers` | The decoder blocks, `Sequence[Layer]` of the family's `Layer`. |
| `model.embed_tokens`, `model.norm`, `model.lm_head` | The embedding, the final norm, the unembedding, as envoys. |
| `model.layers[i].self_attn`, `.mlp`, `.linear_attn` | The family's `Attention`, `Mlp`, `LinearAttention`; `linear_attn` on a hybrid's DeltaNet blocks only, `mlp` absent on OPT. |
| `model.layers[i].input_layernorm`, `.post_attention_layernorm` | Aliases of the block's norms where the family has them; their meaning varies by family (see [families.md](families.md)). |
| `model.vision`, `model.projector` | On a multimodal wrapper whose family names them: the vision tower (a `Vision`; `vision.layers[i]`, `vision.patch_embed`, `vision.norm`; sizes `num_layers`, `hidden_size`, `num_heads`, `head_dim`, `intermediate_size`, `patch_size`, `image_size`; values in [`Vision`](#vision)) and the last module before the scatter into the text stream. |
| `model.add_prefix_false_tokenizer` | The checkpoint's tokenizer loaded with `add_prefix_space=False`, so `"word"` and `" word"` differ; loaded on first use. |
| `model.tokenizer`, `model.config` | nnsight's, unchanged. |

## `Layer`

The decoder block. `Layer.returns_tuple` (class attribute, default `False`) says whether the block returns `(hidden_states, ...)`; GPT-J, GPT-Neo, CodeGen, GPT-NeoX-Japanese, BLOOM, MPT, Falcon, GLM-5, Bamba and Falcon-H1 set it.

| Value | Layout | Assignable | Description |
|---|---|---|---|
| `layer_output` | `Residual` | yes; in-place edits reach the model | The residual stream leaving the block, a tensor even when the block returns a tuple; assigning to a tuple block keeps the other elements. An `EProperty` over `.output`. |

| Method | Signature | What |
|---|---|---|
| `skip_with` | `skip_with(hidden: Tensor) -> None` | Skip this block; `hidden` takes the place of its `layer_output`, as `(hidden, None)` when `returns_tuple`. Inside a trace, before the block runs. |

## `Attention`

A softmax-attention module. The base class reads everything but `attention_output` inside transformers' shared eager attention forward, reached through the module's `attention_interface` call (`INTERFACE = "attention_interface_1"`). GPT-J, GPT-Neo, CodeGen, GPT-NeoX-Japanese, XGLM, BLOOM, MPT and Falcon relocate the same names onto their own operations; [families.md](families.md) has the ops.

| Value | Layout | Base location | Assignable | Availability |
|---|---|---|---|---|
| `attention_output` | `Residual` | the module's `.output`, first tensor | yes; in place reaches the model | always |
| `attention_queries` | `Queries` | argument 1 of `attention_interface_1`, after RoPE | assign; in place except on GPT-2 (split views) and MPT (`chunk`) | `interface_reason` |
| `attention_keys` | `Keys` | argument 2, before `repeat_kv` | as above | `interface_reason` |
| `attention_values` | `Values` | argument 3, before `repeat_kv` | as above | `interface_reason` |
| `attention_scores` | `Pattern` | the input of `nn_functional_softmax_0` inside the interface: scaled and masked | yes; in place reaches the model | `interface_reason` |
| `attention_probabilities` | `Pattern` | the output of `nn_functional_dropout_0` inside the interface: after the softmax, in the model dtype, a sink column dropped | yes; in place reaches the model | `interface_reason` |
| `attention_head_outputs` | `HeadOutputs` | return 0 of `attention_interface_1`, before the output projection | yes; in place reaches the model | `interface_reason` |

Each layout is the alias `.layout` returns (`Attention.attention_keys.layout is Keys`), its axes in the [layouts table](#layouts) and in `.dims`. `qk_head_dim` differs from `head_dim` only under multi-head latent attention (DeepSeek: 192 against 128 on the pinned V3 checkpoint).

`interface_reason(envoy)` calls `envoy.off_interface()`, and the base `off_interface()` is `needs_eager`: `"read inside the eager attention forward, but this model runs 'sdpa'; load with attn_implementation='eager'"`. A family overrides `off_interface` to add its own reason (GPT-2's `reorder_and_upcast_attn`).

| Method / attribute | What |
|---|---|
| `off_interface() -> str \| None` | Why the shared interface does not run on this module, or `None`. |
| `SINK` | `True` on a family whose pattern rows sum to less than one (GPT-OSS). |
| `num_heads` | This block's query heads: the module's `num_heads` / `num_attention_heads` / `n_heads` / `n_head`, else the output projection's input width over `head_dim`. |
| `num_kv_heads` | This block's key/value heads as projected: the module's `num_key_value_heads` / `num_kv_heads` / `kv_heads`, else `num_heads // num_key_value_groups`, else `k_proj`'s width over `qk_head_dim`, else `num_heads`. |
| `head_dim` | This block's value and head-output width: the module's `v_head_dim`, else `head_dim` / `head_size`. |
| `qk_head_dim` | This block's query and key width: the module's `qk_head_dim`, else `head_dim` / `head_size`. |

The four sizes are plain read-only properties, read off the module outside or inside a trace (`model.layers[i].self_attn.num_heads`), so they are the block's own on a model whose blocks differ. A family whose module spells a size another way overrides the property on its subclass; a module the base cannot read raises `NotImplementedError` naming the size.

## `Mlp`

| Value | Layout | Base location | Assignable | Availability |
|---|---|---|---|---|
| `mlp_output` | `Residual` | the module's `.output`, first tensor (a mixture of experts returns router scores beside it) | yes; in place reaches the model (Falcon: through a transform, on a copy) | always where the block has an MLP module; OPT has none, so `support()` lists no `mlp.*` key |

`intermediate_size`, a plain read-only property, is this block's hidden width, one routed expert's on a mixture of experts: the module's `experts.intermediate_dim` / `expert_dim` / `intermediate_size` / `ffn_hidden_size`, else its own `intermediate_size` / `ffn_dim`, else the input width of `down_proj` / `c_proj` / `dense_4h_to_h` / `fc2` / `fc_out` / `w2`. JetMoE's `Mlp` overrides it (`hidden_size` on its module is the experts' width).

## `Moe`

`Moe(Mlp)` in `nnterp/components/moe.py`: a mixture of experts. Each MoE family keys a subclass on its MoE module class (the family's `Mlp` where every MLP is a mixture, a `Moe` class beside its `Mlp` where dense and mixture blocks mix); Gemma-4 and GraniteMoE-Hybrid host it on `layers[i].mlp`, which the block hands the `router` and `experts` envoys. Child names: `router` (aliased from `gate`), `experts`, `shared_experts` (aliased from `shared_expert` / `shared_mlp`). [../usage/mixture-of-experts.md](../usage/mixture-of-experts.md) is the page.

| Value | Layout | Base location | Assignable | Availability |
|---|---|---|---|---|
| `router_logits` | `RouterLogits` | `router.source.F_linear_0.output`: the router's projection, before the scoring (a family's own projection op or module where it differs) | yes, and in place: the router scores what is written | every mixture but Doge's |
| `expert_weights` | `ExpertWeights` | `experts.inputs`, `select=2` | yes, and in place; zero ablates the slot | not on Llama 4 (dense scores) or Doge |
| `expert_indices` | `ExpertIndices` (`Int`) | `experts.inputs`, `select=1` | yes, and in place; reroutes (the weight is not recomputed) | not on Doge |
| `expert_outputs` | `ExpertOutputs` | `experts.source.experts_forward_1.source.weighted_out_view_0.output` | yes, and in place | `needs_grouped_experts`: only under `experts_implementation="grouped_mm"` (the default) or `"batched_mm"`; never on DBRX, JetMoE, Llama 4, Doge, or Nemotron-H with `moe_latent_size` |
| `routed_output` | `Residual` | `experts.output` | yes, and in place | not on Doge |
| `shared_expert_output` | `Residual` | `shared_experts.output` | yes, and in place | `no_shared_expert`: unavailable on a mixture without one |

Every value is a `TokenEProperty`, served as `[batch, seq, ...]` from the model's flat `[batch * seq, ...]` tensor, one invoke's rows under several invokes. `num_experts` and `top_k` are plain read-only properties read off the router or the experts module. `SCORING` is a class attribute naming what the logits mean (`"softmax"`, `"topk_softmax"`, `"sigmoid"`, `"sparsemixer"`; per block on DeepSeek-V4: `"hash"` or the config's `scoring_func`). `no_mixture()` returns why the module runs no mixture, or `None`; every value's availability starts from it (Gemma-4 and GraniteMoE-Hybrid on a dense checkpoint, Doge).

## `RecurrentMixer`

The base of a recurrent mixer's envoy (`nnterp/components/recurrent.py`): how its values are reached at the kernel call its forward makes, apart from what they are. A subclass sets the constants and declares its values at `kernel("inputs")` / `kernel("output")`; the base provides `attention_output`, the per-token state, the availability predicates and `route_kernels`.

| Constant | Default | What |
|---|---|---|
| `BRANCH` | `"use_precomputed_states_0"` | The binding the forward makes before it branches: `True` when the call continues from a cached state. |
| `SEQ_OP` | `"apply_mask_to_padding_states_0"` | The forward's masking of its input, once per call on every mixer: `[batch, seq, ...]`, the call's length. |
| `CHUNK_KERNEL` | `None` | The call a prompt runs through. |
| `RECURRENT_KERNEL` | `None` | The call each decode step of `generate` runs through. |
| `STATE_OP` | `None` | Inside the token-by-token kernel, the binding of the state after each token's update; `None` when the kernels do not materialize it, and then `state`, `states`, `state_after` and `set_state_after` are unavailable. |
| `STEP_STATE_OP` | `None` | Set when the decode kernel is a single-step update rather than the token loop (Mamba-1): the binding of the new state inside it. The prompt's kernel is then the token loop, and `route_kernels(..., "torch")` binds each name to its own pure-torch function. |
| `KERNEL` | a `staticmethod` of the base | Whichever kernel fires on this call, by the forward's own test (`use_precomputed_states and seq_len == 1`), decided once per call (`per_call`): `RECURRENT_KERNEL` when `BRANCH` is true and the sequence axis at `SEQ_OP` is 1, `CHUNK_KERNEL` otherwise, a prompt or several tokens over a cached state. The two bindings are read in the order the forward makes them. |

| Value | Layout | Location | Assignable | Availability |
|---|---|---|---|---|
| `attention_output` | `Residual` | the module's `.output`, first tensor | yes | always |
| `state` | `State` | `STATE_OP` inside the token-by-token kernel: one occurrence per token, walked with `tracer.iter` | yes: the following tokens continue from the write | `needs_recurrent_routing` |
| `states` | `States` | every occurrence of `STATE_OP` in this call, stacked; a `DerivedEProperty` | no (`AttributeError`; use `set_state_after`) | `needs_recurrent_routing` |

`_seq()` is the call's number of tokens, `attention_queries.shape[1]` in the base; a mixer whose kernel is laid out otherwise overrides it.

## `LinearAttention`

A hybrid's gated DeltaNet mixer (`layers[i].linear_attn`), a `RecurrentMixer` that sets the constants and declares eight values; `attention_output`, `state`, `states`, `state_after` and `set_state_after` are the base's. Everything but `attention_output` is read at the delta-rule kernel call, whichever fires on this step.

| Constant | Value | What |
|---|---|---|
| `CHUNK_KERNEL` | `"torch_chunk_gated_delta_rule_0"` | The call a prompt runs through. |
| `RECURRENT_KERNEL` | `"torch_recurrent_gated_delta_rule_0"` | The call each decode step of `generate` runs through. |
| `STATE_OP` | `"last_recurrent_state_3"` | Inside the token-by-token kernel, the binding of the state after each token. |

| Value | Layout | Location | Assignable | Availability |
|---|---|---|---|---|
| `attention_queries` | `LinearQK` | `KERNEL` argument 0 | yes | `needs_torch_kernels` |
| `attention_keys` | `LinearQK` | `KERNEL` argument 1 | yes | `needs_torch_kernels` |
| `attention_values` | `LinearV` | `KERNEL` argument 2 | yes | `needs_torch_kernels` |
| `decays` | `Gates` (`ChannelGates` on Kimi-Linear) | `KERNEL` keyword `g`; float32, non-positive | yes | `needs_torch_kernels` |
| `betas` | `Gates` | `KERNEL` keyword `beta`; in `(0, 1)` | yes | `needs_torch_kernels` |
| `state_input` | `State`, or `None` on a fresh prompt | `KERNEL` keyword `initial_state`, read as a clone of the cache's buffer | yes | `needs_torch_kernels` |
| `attention_head_outputs` | `LinearV` | `KERNEL` return 0, before the gated norm and `out_proj` | yes | `needs_torch_kernels` |
| `state_output` | `State` | `KERNEL` return 1 | yes | `needs_torch_kernels` |

The base's methods, on every `RecurrentMixer`:

| Method | Signature | What |
|---|---|---|
| `state_after` | `state_after(t: int) -> Tensor` | The state after token `t` of this call, counted from the call's own first token. |
| `set_state_after` | `set_state_after(t: int, value: Tensor) -> None` | Assign `state` at token `t`; positions after `t` continue from `value`. Reads follow the forward: read earlier positions before the write, later ones after, and `states` only before it. |

`heads` on these layouts is the module's `num_v_heads`, `key_dim` its `head_k_dim` and `value_dim` its `head_v_dim`. On a decode step the sequence axis is 1.

## `SelectiveScan`

A Mamba-1 mixer (`layers[i].linear_attn` on Mamba, Falcon-Mamba and Jamba's Mamba blocks), a `RecurrentMixer` over `mamba_selective_scan` (a prompt) and `mamba_selective_state_update` (a decode step). The kernel's tensors are channel-first; each value is a tokens-first view of its argument, a write laid back out. `KERNEL` is the base's: the decode kernel when the call is cached and one token long (the forward's condition), the scan otherwise.

| Constant | Value | What |
|---|---|---|
| `CHUNK_KERNEL` | `"mamba_selective_scan_0"` | The scan a prompt runs through; its pure-torch function is the token loop. |
| `RECURRENT_KERNEL` | `"mamba_selective_state_update_0"` | The single-step update each decode step runs through. |
| `STATE_OP` | `"ssm_state_3"` | Inside the scan's token loop, the state after each token. |
| `STEP_STATE_OP` | `"ssm_state_0"` | Inside the decode kernel, the updated state. |
| `STEP_OUTPUT_OP` | `"ssm_state_to_0"` | Inside the decode kernel, the updated state as it is copied into the cache. |
| `READ_OPS` | `("scan_output_5", "out_1")` | Inside the scan and the decode kernel, `y` after the `D` skip and before the gate. |
| `ARGUMENTS` | `{"x": (0, 1), "dt": (1, 2), "A": (2, 3), "B": (3, 4), "C": (4, 5), "dt_bias": ("delta_bias", "dt_bias")}` | Each argument's position or keyword in the scan and in the decode kernel. |

| Value | Layout | Location | Assignable | Availability |
|---|---|---|---|---|
| `attention_queries` | `ScanQK` | `C`: `KERNEL` argument 4 (scan) or 5 (decode), `[batch, seq, 1, state_dim]` | yes | `needs_torch_kernels` |
| `attention_keys` | `ScanQK` | `B`: argument 3 or 4 | yes | `needs_torch_kernels` |
| `attention_values` | `ScanValues` | `x`: argument 0 or 1 | yes | `needs_torch_kernels` |
| `betas` | `ScanSteps` | `softplus(dt + dt_bias)` from the call's arguments; a `DerivedEProperty` | no | `needs_torch_kernels` |
| `decays` | `ScanDecays` | `A * betas`; a `DerivedEProperty` | no | `needs_torch_kernels` |
| `state_input` | `ScanState`, or `None` on a prompt | the decode kernel's argument 0, a clone | on a decode step | `needs_torch_kernels` |
| `attention_head_outputs` | `ScanValues` | `READ_OPS` inside the kernel: `y = C.h + D x` before `silu(z)` and `out_proj` | yes | `needs_kernel_source` |
| `state_output` | `ScanState` | the scan's return 1, or `STEP_OUTPUT_OP` inside the decode kernel | yes: what the cache receives | `needs_kernel_source` |
| `state` / `states` | `ScanState` / `ScanStates` | the base's, at `STATE_OP` or `STEP_STATE_OP` | as the base | `needs_token_loop` |

## `StateSpace`

A Mamba-2 (SSD) mixer (`layers[i].linear_attn` on Mamba-2, Nemotron-H, Bamba and Falcon-H1), a `RecurrentMixer` with SSD's `h = exp(dt·A)·h + dt·x Bᵀ`, `y = h C + D·x` mapped onto the shared names ([../usage/state-space.md](../usage/state-space.md)). Everything but `attention_output` is read at the scan kernel call. The two kernels take their arguments in different places and the decode update has no sequence axis, so each value selects by the kernel that fires (`select=argument(name)`) and a decode step's tensors get a sequence axis of 1 (removed again on assignment).

| Constant | Value | What |
|---|---|---|
| `CHUNK_KERNEL` | `"mamba2_chunk_scan_0"` | The call a prompt runs through. |
| `RECURRENT_KERNEL` | `"mamba2_selective_state_update_0"` | The call a decode step runs through. |
| `STATE_OP` | `None` | Neither kernel binds the state once per token: `state` (a per-occurrence value) is unavailable. |
| `CHUNK_STATES` | `"new_states_0"` | Inside the chunk scan, the state at every chunk boundary, `[batch, chunks + 1, heads, head_dim, state_dim]`: what `states` reads under `chunk_per_token`. |
| `UPDATED_STATE` | `"ssm_states_0"` | Inside the update, the new state before it is copied into the cache. |
| `CHUNK_ARGUMENTS`, `RECURRENT_ARGUMENTS` | name -> position or keyword | Where each kernel's call site passes `hidden_states`, `dt`, `A`, `B`, `C`, `dt_bias` and the state. |

| Value | Layout | Location | Assignable | Availability |
|---|---|---|---|---|
| `attention_queries` | `SSDQueries` | `C`: argument 4 of the scan, 5 of the update | yes | `needs_torch_kernels` |
| `attention_keys` | `SSDKeys` | `B`: argument 3 of the scan, 4 of the update | yes | `needs_torch_kernels` |
| `attention_values` | `SSDValues` | `x` (`hidden_states`): argument 0 of the scan, 1 of the update | yes | `needs_torch_kernels` |
| `betas` | `Gates` | `dt`: argument 1 of the scan, 2 of the update, read as `softplus(dt + dt_bias)`, clamped to `dt_limit` on a prompt | yes: written back as `dt = log(expm1(betas)) - dt_bias`; positive; clamped by the scan on a prompt | `needs_torch_kernels` |
| `decays` | `Gates` | `dt`, read as `A * betas`; float32, non-positive | yes: written back as the `betas` it implies, `decays / A` | `needs_torch_kernels` |
| `state_input` | `State`, or `None` on a fresh prompt | the scan's keyword `initial_states`, the update's argument 0; a transposed clone | yes | `needs_torch_kernels` |
| `attention_head_outputs` | `SSDHeadOutputs` | `y`: the scan's return 0 (the whole return without a cache), the update's return | yes | `needs_torch_kernels` |
| `state_output` | `State` | the scan's return 1; inside the update, `UPDATED_STATE`; transposed | yes | `needs_torch_kernels`; `Unavailable` at the read on a prompt run with `use_cache=False` |
| `states` | `States` | inside the scan, `CHUNK_STATES[:, 1:]`, transposed (a copy); on a decode step `state_output` with a sequence axis of 1; a `DerivedEProperty` | no (`AttributeError`) | `needs_per_token_chunks` |
| `state` | | `unavailable(...)`: the scan has no per-token occurrence of the state | no | never: `"the chunk scan computes every token's state in one tensor per call, ..."` |
| `set_state_after` | | `unavailable(...)` in place of the base's method, so `support()` lists it | no | never: `"the chunk scan computes every boundary state in one cumulative step from the initial state, ..."` |

`state_after(t)` is `states[:, t]`, and raises `Unavailable` with `needs_per_token_chunks`' reason without `chunk_per_token`. `KERNEL` is the base's: the update when the call is cached and one token long, the chunk scan otherwise. The kernel's arguments are read from the model once per call (`_arguments`, through `per_call`), on the call's first need, so any forward-order combination of values reads in one trace, and a value that needs another argument (`betas` needs `dt_bias`) finds it after the model has moved into the kernel.

`heads` is the module's `num_heads`, `groups` its `n_groups`, `head_dim` its `head_dim`, `state_dim` (the state's `key_dim`) its `ssm_state_size`; the state's `value_dim` is `head_dim`.

## `Vision`

`model.vision`, a multimodal wrapper's vision tower, on a family that names it ([../usage/vision.md](../usage/vision.md) lists the towers and wrappers); its blocks are `VisionLayer`, `VisionAttention`, `VisionMlp` (the text components with the stream values annotated `Patches`; a tower that runs on `[patches, vision_hidden]` is served with a leading images axis of 1). `PixtralVision` reads `patch_embeddings` at the packed row entering `ln_pre`. The Qwen ViT's root is a `QwenVision` (its sizes add `spatial_merge_size` and `window_size`) and its attention a `QwenVisionAttention`: the queries, keys and values whole before the per-image split, the head outputs at the concatenation after it, and the scores and pattern unavailable (`PER_IMAGE`). The two image values are the tower's although neither is read inside it: their keys are anchored at the model's root (`"/inputs"`, and the scatter, `"/model.source.inputs_embeds_masked_scatter_0.inputs"` on a Llava wrapper, `"/source.inputs_embeds_masked_scatter_0.inputs"` on Llama 4, whose family names `ROOT_SCATTER`), walked from `vision.root`. Every tower value is unavailable with `no_images()`'s reason on a load no image reaches.

| Value | Layout | Read at | Assignable | Unavailable when |
|---|---|---|---|---|
| `image_token_mask` | `ImageTokenMask` | the model's inputs (`"/inputs"`), so first in the invoke: `input_ids == config.image_token_id` | no (`AttributeError`: assign `input_ids`) | `no_images()`, or the config names no `image_token_id` |
| `patch_embeddings` | `Patches` | `patch_embed.output`, one row per patch before any position embedding, CLS token or pre-norm (a convolution's grid flattened, a view); on `PixtralVision` the packed row entering `ln_pre` | yes | `no_images()` |
| `tower_output` | `Patches` | the last block's stream after `vision.norm` where there is one, before any pooling, CLS dropping or adapter: the tower's `last_hidden_state`, or where the stream is on a tower that returns something after those (Llama 4: `layernorm_post`; Gemma 4: the encoder) | yes | `no_images()` |
| `image_features` | `ImageFeatures` | the features argument of the scatter in the wrapper model's forward (`ImageScatter.scatter`, `.scatter_argument`; the root's own forward where the family names `ROOT_SCATTER`) flat over the batch's image tokens: `layers[0].input[image_token_mask] == image_features` | yes; in-place edits land | as `image_token_mask`, or the family keys no `ImageScatter` on the wrapper's model |

| Member | Signature | What |
|---|---|---|
| `no_images` | `no_images() -> str \| None` | Why no image reaches the model, or `None`: the family names no `projector`, or the load has no processor (a `task="text-generation"` load of a wrapper). The one place that decides. |
| `support` | `support(layer: int \| None = None) -> dict[str, Any]` | The tower's values plus its block values over `vision.layers`, as `StandardizedTransformer.support` reports the text blocks'; with `layer`, that block's. Empty when `no_images()` gives a reason. `model.support()` carries the same rows prefixed `vision.`. |
| sizes | `num_layers`, `hidden_size`, `num_heads`, `head_dim`, `intermediate_size`, `patch_size`, `image_size` | Read off the tower's own config; `image_size` raises `Unavailable` on a tower with no fixed resolution (the Qwen ViT, Pixtral, Gemma 4, Gemma 4 unified). |

## `Standard`

The base of the hosts.

| Member | Signature | What |
|---|---|---|
| `values` | `classmethod values() -> dict[str, EProperty]` | This class's standard values by name, base classes first. |
| `support` | `support() -> dict[str, str \| None]` | Each value here: `None` when available on this envoy, else the reason. |
| `parent`, `root` | nnsight's `Envoy.parent` / `Envoy.root` | The envoy holding this one, and the model envoy at the top. What a value reads its block, or the model's config, processor and family, through (`Vision`'s image values, the Granite mixtures' `residual_multiplier`). |
| `sourced` | `sourced: bool = False` (class attribute) | `True` on a subclass instruments the envoy's forward when it is built and again when real weights replace meta ones, for a value in that forward read after the call has started (Llama 4's `Layer`, whose `Mlp.mlp_output` follows `attention_output`). A path declares where a value is; this flag is what makes such a read serve rather than raise `OutOfOrderError`. |

## `nnterp.families`

| Name | Signature | What |
|---|---|---|
| `lookup` | `lookup(model_type: str) -> ModuleType` | The family for `model_type`: a registered one, else `nnterp.families.<model_type>`, imported on first use. Raises `UnsupportedFamily` when there is neither. |
| `register` | `register(family: ModuleType, *model_types: str) -> ModuleType` | Add a family (any module or object with `RENAME`, `ENVOYS`, and a function per root size it spells its own way) under `model_types`, or, with none given, the type its `__name__` ends in (`TypeError` for a nameless `SimpleNamespace`); consulted before the shipped modules, so it also overrides a shipped family. Returns `family`. |
| `known` | `known() -> list[str]` | The shipped families' model types: the module names in the package (98), alphabetical. |
| `all_families` | `all_families() -> list[ModuleType]` | Every shipped family, imported. For tooling and tests. |
| `REGISTRY` | `dict[str, ModuleType]` | `model_type -> family` for what `register` added. |
| `UnsupportedFamily` | `ValueError` subclass | No module of that name and nothing registered. |
| `nnterp.families.<model_type>` | module attribute | The family module, imported on first access (`nnterp.families.qwen3_5_text`). |

A family module, named after the `model_type` it covers, declares `RENAME: dict[str, str]`, `Layer`, `Attention`, `Mlp` (and `LinearAttention` on a DeltaNet hybrid, `SelectiveScan` on a Mamba-1 mixer, `StateSpace` on a Mamba-2 mixer; a pure state-space family such as Mamba or Mamba-2 has no `Attention` or `Mlp`) subclassing `nnterp.components`'s, and `ENVOYS: dict[type, type]` keying them on its transformers module types. It may also define a module-level function named after any root size, `def <size>(model) -> int`, which the root's `StandardizedProperty` calls in place of its plain rule (`falcon.num_kv_heads`, `deepseek_v2.head_dim`, `gpt2.intermediate_size`), and likewise `def project_on_vocab(model, hidden)`, which its `StandardizedCapability` binds in place of the root's norm, head and softcap (`cohere.project_on_vocab`, `granite.project_on_vocab`).

## `nnterp.components`

### Descriptors

| Descriptor | Signature | What |
|---|---|---|
| `EProperty` | `EProperty(key=None, description=None, unavailable=None, select=None)` | nnsight's `eproperty` plus availability and a path for a key. `key` is a path from the host envoy, dotted segments ending in `output`, `input` or `inputs`: `"output"` is the host's own output; a leading `/` walks from the model's root instead of the host (`envoy.root`), aliases included (`"/inputs"`); a leading `../` (repeatable) steps to the parent by native name; another segment is a child module (aliases included) or, after a `source` segment, an operation; `source` drills into the current module's or operation's forward, instrumenting it for this run (`"source.attention_interface_1.source.nn_functional_softmax_0.output"`, `"../post_attention_layernorm.output"`, `"embed_tokens.output"`, `"../source.hidden_states_view_0.output"`). A function of the envoy returning such a path is allowed (a `RecurrentMixer`'s `kernel("inputs")`, Falcon's `by_alibi`). `None` means the attribute name. `select` picks an element: with `inputs` an int is a positional argument and a str a keyword, with `output` an int indexes the returned tuple; `input` is the call's first argument; a function of the envoy returning one of those (or `None`, the whole value) selects per access, before the served read (`SelectiveScan`'s `_argument(name)` and `StateSpace`'s `argument(name)`, whose two kernels take an argument in different places). A write with `select` (or on `input`) repacks the element and writes the whole value back. `unavailable` is a reason string, or a function of the envoy returning one or `None`, checked on every read and write; `reason(obj)` returns it. `path(obj)` and `inside_forward()` describe the key. A path is walked before every read or write; a missing op raises nnsight's `SourceNotAvailable`. |
| `TokenEProperty` | `TokenEProperty(key=None, description=None, unavailable=None, select=None)` | An `EProperty` (in `nnterp/components/tokens.py`) for a value the model may hold flat over tokens, `[batch * seq, ...]`, one rank below its layout: `__get__` reads as `EProperty` does and serves this invoke's `[batch, seq, ...]` rows of the result, a view, so in-place edits land; `__set__` splices the assigned rows into the whole tensor the location holds, then writes as `EProperty` does. A tensor already at the layout's rank passes through both ways; preprocess, postprocess and transform see the model's own tensor. `Moe`'s routing values use it. |
| `DerivedEProperty` | `DerivedEProperty(compute, description=None, unavailable=None)` | `compute(envoy)` runs at read time inside the trace over any served values; read-only (assignment raises `AttributeError`). Its layout is read off `compute`'s return annotation. |

Every descriptor exposes `.layout` (the layout alias the defining function's return annotation names, or `None`), `.dims` (its axis names as a tuple), `.description`, `.reason(envoy)`. `str(value)` is its line in the envoy's repr, `(name) -> Layout [axes]: description` when the value has a layout (nnsight prints it from #737 on).

### Layouts

The thirty-five `jaxtyping` types every standard value is annotated with, each defined in the file of the envoy that serves it. `nnterp.components` re-exports the thirty-two envoy-level names; the root's three are importable from `nnterp.standardized` only. `value.layout` is the alias itself (`Attention.attention_probabilities.layout is Pattern`), `value.dims` its axes split; a redefinition in a family and a value of your own annotate with the same name (`-> Residual`). `isinstance(tensor, Pattern)` checks rank and dtype.

| Name | Axes | Defined in | Carried by |
|---|---|---|---|
| `Residual` | `batch seq hidden` | `components/layer.py` | `layer_output`, `attention_output`, `mlp_output`, `token_embeddings` |
| `Streams` | `batch seq streams hidden` | `components/layer.py` | `layer_output` on a hyper-connection family (DeepSeek-V4) |
| `StreamWeights` | `batch seq streams` | `components/layer.py` | a hyper-connection family's `attention_post`, `mlp_post` |
| `StreamMixing` | `batch seq streams streams` | `components/layer.py` | a hyper-connection family's `attention_comb`, `mlp_comb` |
| `Logits` | `batch seq vocab` | `standardized.py` | `logits` |
| `NextTokenProbs` | `batch vocab` | `standardized.py` | `next_token_probs` |
| `Tokens` | `batch seq` (`Int`) | `standardized.py` | `input_ids`, `attention_mask` |
| `Queries` | `batch heads seq qk_head_dim` | `components/attention.py` | `attention_queries` |
| `Keys` | `batch kv_heads seq qk_head_dim` | `components/attention.py` | `attention_keys` |
| `Values` | `batch kv_heads seq head_dim` | `components/attention.py` | `attention_values` |
| `Pattern` | `batch heads query key` | `components/attention.py` | `attention_scores`, `attention_probabilities` |
| `HeadOutputs` | `batch seq heads head_dim` | `components/attention.py` | `attention_head_outputs` |
| `LinearQK` | `batch seq heads key_dim` | `components/linear_attention.py` | `linear_attn.attention_queries`, `linear_attn.attention_keys` |
| `SSDQueries` / `SSDKeys` | `batch seq groups state_dim` | `components/state_space.py` | a Mamba-2 `linear_attn.attention_queries` (`C`) / `attention_keys` (`B`) |
| `SSDValues` / `SSDHeadOutputs` | `batch seq heads head_dim` | `components/state_space.py` | a Mamba-2 `linear_attn.attention_values` (`x`) / `attention_head_outputs` (`y`) |
| `LinearV` | `batch seq heads value_dim` | `components/linear_attention.py` | `linear_attn.attention_values`, `linear_attn.attention_head_outputs` |
| `Gates` | `batch seq heads` | `components/linear_attention.py` | `decays`, `betas` |
| `ChannelGates` | `batch seq heads key_dim` | `components/linear_attention.py` | `decays` on Kimi-Linear |
| `State` | `batch heads key_dim value_dim` | `components/recurrent.py` | `state_input`, `state_output`, `state` |
| `States` | `batch seq heads key_dim value_dim` | `components/recurrent.py` | `states` |
| `ScanQK` | `batch seq groups state_dim` | `components/selective_scan.py` | a Mamba-1 `attention_queries`, `attention_keys` |
| `ScanValues` | `batch seq channels` | `components/selective_scan.py` | a Mamba-1 `attention_values`, `attention_head_outputs` |
| `ScanSteps` | `batch seq channels` | `components/selective_scan.py` | a Mamba-1 `betas` |
| `ScanDecays` | `batch seq channels state_dim` | `components/selective_scan.py` | a Mamba-1 `decays` |
| `ScanState` | `batch channels state_dim` | `components/selective_scan.py` | a Mamba-1 `state_input`, `state_output`, `state` |
| `ScanStates` | `batch seq channels state_dim` | `components/selective_scan.py` | a Mamba-1 `states` |
| `RouterLogits` | `batch seq experts` | `components/moe.py` | `router_logits` |
| `ExpertWeights` | `batch seq top_k` | `components/moe.py` | `expert_weights` |
| `ExpertIndices` | `batch seq top_k` (`Int`) | `components/moe.py` | `expert_indices` |
| `ExpertOutputs` | `batch seq top_k hidden` | `components/moe.py` | `expert_outputs` |
| `Patches` | `images patches vision_hidden` | `components/vision.py` | a tower block's `layer_output`, `attention_output`, `mlp_output`; `vision.patch_embeddings`, `vision.tower_output` |
| `ImageTokenMask` | `batch seq` (`Bool`) | `components/vision.py` | `vision.image_token_mask` |
| `ImageFeatures` | `image_tokens hidden` | `components/vision.py` | `vision.image_features` |

The axis names are the same on every layout (`batch` axis 0 everywhere, `seq` the token axis, `heads` the query heads, `kv_heads` the key/value heads, `head_dim` the width of values and head outputs, `qk_head_dim` that of queries and keys, `query`/`key` a pattern's two token axes, `key_dim`/`value_dim` a DeltaNet state's two sides, `channels`/`state_dim` a Mamba-1 state's, `groups` the `B`/`C` groups, `experts` a router's classes, `top_k` the routing slots); the comments above each alias state them, and [../usage/layouts.md](../usage/layouts.md) is the page.

### Availability predicates and constants

| Name | Signature / value | What |
|---|---|---|
| `unavailable` | `unavailable(reason: str) -> EProperty` | A value a family does not have: assign it in the class body in place of the inherited one. The name stays in the tree and the repr (`"Unavailable: <reason>"`), and every access raises `Unavailable`. |
| `needs_eager` | `needs_eager(envoy) -> str \| None` | The reason when `config._attn_implementation != "eager"`. |
| `interface_reason` | `interface_reason(envoy) -> str \| None` | `envoy.off_interface()`: the predicate of the base `Attention`'s interior values. |
| `needs_torch_kernels` | `needs_torch_kernels(envoy) -> str \| None` | The reason when the mixer's `CHUNK_KERNEL` or `RECURRENT_KERNEL` name dispatches to an optimized kernel (`flash-linear-attention`, `mamba_ssm`) with no Python source. |
| `needs_per_token_chunks` | `needs_per_token_chunks(envoy) -> str \| None` | `needs_torch_kernels`, then, when the mixer's `chunk_size` is not 1, the reason naming the chunk size and `chunk_per_token`: the predicate of `StateSpace.states`. |
| `needs_recurrent_routing` | `needs_recurrent_routing(envoy) -> str \| None` | The reason when `STATE_OP` is `None` (the kernels do not materialize the state per token), then `needs_torch_kernels`, then the reason when the prompt kernel is not the token-by-token loop (or, with `STEP_STATE_OP`, the decode kernel is still the dispatcher): call `route_kernels`. |
| `needs_kernel_source` | `needs_kernel_source(envoy) -> str \| None` | `needs_torch_kernels`, then the reason when either kernel name is still transformers' dispatcher, whose body is not the kernel's: a `SelectiveScan` value read inside a kernel (`attention_head_outputs`, `state_output`). |
| `needs_token_loop` | `needs_token_loop(envoy) -> str \| None` | `needs_recurrent_routing`, then the reason when a Mamba-1 checkpoint sets `use_mambapy` with `mambapy` installed (a parallel scan, no per-token binding). |
| `needs_grouped_experts` | `needs_grouped_experts(envoy) -> str \| None` | `no_mixture()`, then the reason when the experts run neither `grouped_mm` nor `batched_mm` (it names `experts_implementation=`): the predicate of `Moe.expert_outputs`. |
| `mixture_reason` | `mixture_reason(envoy) -> str \| None` | `envoy.no_mixture()`: the predicate of `Moe`'s other routing values. |
| `no_shared_expert` | `no_shared_expert(envoy) -> str \| None` | `no_mixture()`, then `"this mixture has no shared expert"` when the module has no `shared_experts` / `shared_expert` / `shared_mlp` child. |
| `LOGITS`, `DISPATCH`, `PER_SLOT` | `"F_linear_0"`, `"experts_forward_1"`, `"weighted_out_view_0"` | The router's logits op, the call inside transformers' `@use_experts_implementation` wrapper, and the per-slot view inside its `grouped_mm` / `batched_mm` forwards. |
| `INTERFACE` | `"attention_interface_1"` | The shared attention call every interface family makes. |
| `NOT_ON_INTERFACE` | `"The attention does its own arithmetic rather than transformers' shared attention interface; not mapped for this family yet"` | The reason for `unavailable(NOT_ON_INTERFACE)` in a family that has not mapped an interior value onto its own arithmetic. No shipped family uses it: all four own-arithmetic families map every interior value. |

### Helpers

| Name | Signature | What |
|---|---|---|
| `per_call` | `per_call(envoy, key: str, compute: Callable[[], Any]) -> Any` | `compute()` once per call of the envoy's module. The record, `(call, value)` per `(envoy.path, key)`, lives on the worker (the greenlet running the intervention code): each run of each invoke has its own, a replayed `model.edit` included, and nothing stays on the envoy. One rule names the call: the step a read is pinned to by `tracer.iter`, and, relaxed, on step 0 or outside `tracer.iter`, how many times the module's `.output` has been passed (inside call c that is c; between calls, the one about to start). Defined in `nnterp.components.recurrent`. |
| `pinned` | `pinned(n: int \| None)` | A context manager that pins the worker's reads to occurrence `n` of their location, as `tracer.iter[n]` does (`None` relaxes the pin), and restores the pin the worker had on the way out. A pinned read relaxes the pin, so one read per `with`. What `states`, `state_after` and `set_state_after` read each token with. Defined in `nnterp.components.recurrent`. |
| `route_kernels` | `route_kernels(family, kernel: str = "torch") -> None` | Bind a family's recurrent kernel names process-wide: `"torch"` to transformers' pure-torch kernels (on a mixer with a `STATE_OP`, the prompt's name to the token-by-token loop, the only kernel that materializes `state`, `states`, `state_after`, `set_state_after`: the decode kernel's function on a gated DeltaNet, so both names; the scan's own on Mamba-1, where each name keeps its own), `"default"` back to what the modeling module bound at import. `family` is `model.family`, `nnterp.families.<name>` or the modeling module. Call it before tracing a layer. |
| `route_delta_rule` | `route_delta_rule(family, kernel: str = "recurrent") -> None` | `route_kernels` in the delta rule's words: `"recurrent"` is `"torch"`, `"chunked"` is `"default"`. |
| `chunk_per_token` | `chunk_per_token(model, enabled: bool = True) -> None` | Set every `StateSpace` mixer's module `chunk_size` to 1, so the chunk scan's boundaries are the tokens and `states` / `state_after` read the state after each; `enabled=False` restores the chunk size each mixer was built with (`config.chunk_size`, or `mamba_chunk_size`). Per model, unlike `route_kernels`; slower on long prompts (the inter-chunk recurrence is quadratic in the number of chunks). `ValueError` on a model with no `StateSpace`. |
| `seq_first` | `seq_first(value: Tensor) -> Tensor` | `value.transpose(1, 2)`: `[batch, heads, seq, d]` to `[batch, seq, heads, d]` as a view, and its own inverse. Used by families whose arithmetic keeps heads first. |
| `first_tensor` | `first_tensor(value) -> Tensor` | The first element of a tuple output, or the tensor itself. |
| `rewrap` | `rewrap(envoy, value: Tensor) -> Any` | `value` back in the module's current output tuple, if any. |
| `rows`, `splice` | `rows(value, rank) -> Any`, `splice(whole, value, rank) -> Any` | A tensor flat over tokens as this invoke's `[batch, seq, ...]` view; this invoke's rows written back into the whole flat tensor. What `TokenEProperty` reads and writes through. Defined in `nnterp.components.tokens`. |
| `module_int`, `in_width` | `module_int(module, *names) -> int \| None`, `in_width(module, *names) -> int \| None` | The first of `names` the module holds as an integer; the input width of the first of `names` it has as a projection (`nn.Linear` or `Conv1D`). What the per-module sizes read with. Defined in `nnterp.components.standard`. |

## `nnterp.prompt_utils`

Prompt helpers on the standard values.

| Name | Signature | What |
|---|---|---|
| `get_first_tokens` | `get_first_tokens(words: str \| list[str], model_or_tokenizer, use_hacky_implementation: bool = False) -> list[int]` | The first token of `word` and of `" word"` for each word, deduplicated. Given a model, uses `model.add_prefix_false_tokenizer`; a tokenizer that adds a prefix space falls back to tokenizing `"🍐word"` and dropping the pear, or raises `TokenizationError`. |
| `Prompt` | `@dataclass Prompt(prompt: str, target_tokens: dict[str, list[int]], target_strings: dict \| None = None)` | A prompt with named sets of target tokens. |
| `Prompt.from_strings` | `Prompt.from_strings(prompt: str, target_strings: dict[str, str \| list[str]] \| list[str] \| str, model_or_tokenizer) -> Prompt` | Build from words; a string or list is one target named `"target"`. |
| `Prompt.has_no_collisions` | `has_no_collisions(ignore_targets: str \| list[str] \| None = None) -> bool` | Whether no token id belongs to two targets. |
| `Prompt.get_target_probs` | `get_target_probs(probs: Tensor, layer: int \| None = None) -> dict[str, Tensor]` | Each target's mass from `probs` of shape `[batch, layers, vocab]`. |
| `Prompt.run` | `run(model, get_probs: Callable) -> dict[str, Tensor]` | `get_probs(model, prompt)` reduced to each target's mass. |
| `next_token_probs_unsqueeze` | `next_token_probs_unsqueeze(model, prompt, remote: bool = False, **_) -> Tensor` | `compute_next_token_probs` with a layer axis of one, `[batch, 1, vocab]`; the default `get_probs_func`. |
| `run_prompts` | `run_prompts(model, prompts: list[Prompt], batch_size: int = 32, get_probs_func: Callable \| None = None, func_kwargs: dict \| None = None, remote: bool = False, tqdm=None) -> dict[str, Tensor]` | Each target's probability mass per prompt, `[num_prompts, layers]`; all prompts must name the same targets. |
| `TokenizationError` | `Exception` subclass | A word could not be tokenized as a standalone first token. |

## `nnterp.nnsight_utils`

Activation helpers on the standard values. `GetActivations = Callable[[StandardizedTransformer, int], Tensor]`; the default is `layer_output(model, layer)`, the residual stream leaving block `layer`.

| Name | Signature | What |
|---|---|---|
| `layer_output` | `layer_output(model, layer: int) -> Tensor` | `model.layers[layer].layer_output`. |
| `get_token_activations` | `get_token_activations(model, prompts=None, layers=None, get_activations=None, remote=False, idx=None, tracer=None) -> Tensor` | `[num_layers, num_prompts, hidden]` at one position (`idx`, default `-1`) of every prompt; a negative index needs left padding, a positive one right padding. With `tracer`, reads inside the caller's open trace. |
| `collect_last_token_activations_session` | `collect_last_token_activations_session(model, prompts: list[str], batch_size: int, layers=None, get_activations=None, remote=False, idx=None) -> Tensor` | The same over batches inside one `model.session`, so a remote run is one request. |
| `collect_token_activations_batched` | `collect_token_activations_batched(model, prompts: list[str], batch_size: int, layers=None, get_activations=None, remote=False, idx=None, tqdm=None, use_session: bool = True) -> Tensor` | The same over batches; a remote run goes through the session variant unless `use_session=False`. |
| `compute_next_token_probs` | `compute_next_token_probs(model, prompt: str \| list[str], remote: bool = False) -> Tensor` | `model.next_token_probs` per prompt, `[num_prompts, vocab]` on the CPU. |

## Exceptions

| Exception | Raised when |
|---|---|
| `nnterp.Unavailable` (`RuntimeError`) | A standard value this checkpoint does not have is read or written: `"<path>.<name> is not available: <reason>"`, at that line, before the model runs. `support()` gives the same reason without raising. `hasattr(envoy, name)` also raises it. |
| `nnterp.UnsupportedFamily` (`ValueError`) | The checkpoint's `model_type` has no family module and nothing registered; the message lists the known types. |
| `nnsight.intervention.source.SourceNotAvailable` | An `EProperty`'s path names an operation that is not under `.source` in this run: the forward took a path the family does not expect. |
| `nnterp.prompt_utils.TokenizationError` | A word has no standalone first token under the tokenizer. |
| `AttributeError` | Assigning a read-only value: `next_token_probs`, `input_size`, `states`, or any `DerivedEProperty`. |
| `nnsight.intervention.interleaver.OutOfOrderError` | A value is read after the model ran past its location: a block's interior after its output, `input_ids` after a block, Falcon's queries before its values. |

## Gotchas

- Nothing assigned inside a trace survives it unless it is `.save()`d, and the save is bound to a name: `x = model.logits.save()`.
- Reads follow forward order within one invoke: the model's input (`input_ids`, `attention_mask`, `input_size`) first, a block's interior (`attention_probabilities`, queries, scores) before that block's `attention_output`, block 3 before block 5. On Falcon read `attention_values` before `attention_queries` / `attention_keys`; on DeltaNet read `states` before any state write.
- Decide which blocks have `self_attn` outside the trace: `[i for i, l in enumerate(model.layers) if getattr(l, "self_attn", None) is not None]`. Compare with `is not None`: an envoy has no truthiness (`getattr(layer, "self_attn", None) or layer.linear_attn` raises `TypeError: object of type 'LlamaAttention' has no len()`), and inside a trace `getattr(envoy, name, None)` can trip a served value.
- `hasattr(envoy, "attention_probabilities")` raises `Unavailable` when the value is unavailable; ask `support()` instead.
- The interior attention values need `attn_implementation="eager"`, which the constructor does not force; BLOOM and MPT are the exception (their pattern is their own dropout and carries no `attn_implementation` predicate).
- `get_topk_closest_tokens(hidden)` takes a residual-stream tensor and projects it itself; passing `project_on_vocab`'s output fails inside `norm` with a shape error.
- GPT-2's and MPT's queries, keys and values are views of one fused tensor: assign, do not edit in place. Falcon's `mlp_output` is a copy; assignment and in-place edits reach the model through a transform.
- `model.logits` is the output's `.logits` (softcapped on Gemma-2, scaled on Cohere and Granite); `model.lm_head.output` is the raw projection, and `model.project_on_vocab(model.layers[-1].layer_output)` is `logits`.
- `next_token_probs`, `input_size` and `states` are read-only.
- `route_kernels(model.family, "torch")` before tracing a DeltaNet layer whose `state` you want; a forward `.source` has already instrumented keeps the binding it was compiled with.
- `import nnterp` (or `nnsight`) before any `transformers.models...modeling_*` import; the reverse order segfaults at import on this stack.

## Related

- [families.md](families.md): what each family relocates and what it lacks.
- [glossary.md](glossary.md): the vocabulary these tables use.
- [../usage/root-values.md](../usage/root-values.md), [../usage/methods.md](../usage/methods.md), [../usage/availability.md](../usage/availability.md), [../usage/layouts.md](../usage/layouts.md), [../usage/attention-interior.md](../usage/attention-interior.md), [../usage/delta-net.md](../usage/delta-net.md).
- [../extending/custom-values.md](../extending/custom-values.md) and [../developing/eproperty-internals.md](../developing/eproperty-internals.md) for the descriptors in depth.
- nnsight `docs/reference/api-quick-reference.md` for `trace`, `generate`, `session`, `tracer.iter`, `.source`, `remote=`.
