# nnterp 1.x vs nnterp — gap report, 2026-09-29

Read-only comparison of nnterp 1.x (the renaming-based `StandardizedTransformer`, branch `internals-accessors` at `2db11d8`, 2026-09-25) against nnterp, this package (2026-09-29). Both import in one environment (transformers 5.17.0, nnsight 0.8.0). Line references into nnterp 1.x files are at `2db11d8`; line references into this package are at the time of the comparison and drift as it changes. Claims marked *(ran)* were checked by executing code; everything else is from reading.

Abbreviations: `st.py` = nnterp 1.x `nnterp/standardized_transformer.py`, `ru.py` = nnterp 1.x `nnterp/rename_utils.py`, `S.py` = `nnterp/standardized.py`, `comp/` = `nnterp/components/`, `fam/` = `nnterp/families/`.

## 1. Surface map

Status legend: **same** (name and semantics), **renamed** (same semantics, new spelling), **changed** (an nnterp 1.x user would get a different result), **missing** (no nnterp equivalent), **dropped** (nnterp's `docs/developing/contributing.md:159-177` lists it as a design choice).

### 1a. Package entry (`nnterp/__init__.py`)

| nnterp 1.x | nnterp equivalent | status | note |
|---|---|---|---|
| `StandardizedTransformer` (`__init__.py:5`) | `nnterp.StandardizedTransformer` (`S.py:70`) | changed | Same class name; constructor, accessors and validation all differ (rows below). |
| `StandardizedVLM` (`:5`) | none | missing / dropped | `S.py:137` hardcodes `task="text-generation"`; `contributing.md:194-196` lists VLMs as open. |
| `load_model(model, use_vllm, allow_experimental_vllm, text_only, **kw)` (`:21-66`) | none | missing | nnterp has one entry point, the class. |
| `detect_automodel(model, trust_remote_code, text_only)` (`utils.py:90-179`) | none | missing | nnterp reads `AutoConfig` in `_read_config` (`S.py:450-466`) only to pick the family. |
| `get_rename_dict(rename_config)` (`ru.py:239-269`) | `family.RENAME` (`fam/*.py`), `nnterp.families.lookup(model_type)` (`fam/__init__.py:44`) | changed | One dict per family, not one global list. |
| `ModuleAccessor(model, rename_config, rename)` (`nnsight_utils.py:203-256`) | none | missing | Raw-`nn.Module` access by standard name has no nnterp counterpart. |

### 1b. Constructor kwargs

| nnterp 1.x `StandardizedTransformer(...)` (`st.py:675-687`) | nnterp (`S.py:128-136`) | status | note |
|---|---|---|---|
| `model: str \| Module` | `repo_id: Any` (str or Module; `S.py:458-459`) | same | Positional name differs. |
| `check_renaming=True` | none | dropped | Nothing runs at construction (`contributing.md:174-175`). |
| `remote=False` (sets `allow_dispatch=False`, scan-only checks; `st.py:168-176`) | not a nnterp kwarg; `**kwargs` passes it to nnsight, which accepts it and keeps the model on meta *(ran: `dispatched=False`)* | changed | No nnterp-side behavior attaches to it; `remote=True` on `trace()` is the contract (`docs/usage/remote.md`). |
| `allow_dispatch=True` | none | dropped | No scan/trace fallback machinery. |
| `enable_attention_probs=False` (forces `attn_implementation="eager"`, refuses other values; `st.py:318-324`) | none: pass `attn_implementation="eager"` yourself | changed | See §3.2. |
| `check_attn_probs_with_trace=None` | none | dropped | Causal write check lives in the suite (`tests/families/suite.py:303-317`). |
| `rename_config: RenameConfig` (`ru.py:53-156`) | `rename=`, `envoys=`, `nnterp.families.register()` | changed | See `RenameConfig` rows in §1h. |
| `automodel=None` | none | missing | |
| `text_only=False` | none | missing | |
| `tokenizer_kwargs=None` (`st.py:717-718`) | `tokenizer_kwargs=None` (`S.py:154-155`) | same | setattr on the tokenizer, both. |
| `rename=` via `**kwargs` (`st.py:326`) | `rename=` (`S.py:132`, merged over `family.RENAME` `:146`) | same | |
| `device_map="auto"` default (`st.py:316`) | none: nnsight's default | changed | Pass `device_map="auto"` explicitly. `contributing.md:216`. |
| `**kwargs` → `TransformersModel` with `task="text-generation"` (`st.py:710-716`) | same, `task` defaults to `"text-generation"` (`S.py:137`) | same | |
| — | `envoys=` (`S.py:133`, merged over `family.ENVOYS` and tp envoys `:147-151`) | new | |

### 1c. Published ints / model attributes

| nnterp 1.x (`st.py:111-127`, set `:220-252`) | nnterp | status | note |
|---|---|---|---|
| `num_layers` | `num_layers` (`S.py:399-401`) | same | |
| `attention_layers: list[int]` (`ru.py:428-439`) | none | missing | `[i for i,l in enumerate(model.layers) if getattr(l,"self_attn",None) is not None]` (`api-quick-reference.md:329`); `contributing.md:216-218` marks trivial. |
| `linear_attention_layers: list[int]` | none | missing | as above with `linear_attn`. |
| `num_heads` (keys `n_heads/num_attention_heads/n_head/num_heads`, `ru.py:175-176`) | `num_heads` = `config.num_attention_heads` (`S.py:411-413`) | same | transformers' `attribute_map` covers GPT-2/BLOOM/MPT/DBRX *(ran)*. |
| `hidden_size` (`hidden_size/d_model/n_embd`) | `hidden_size` = `config.hidden_size` (`:379-381`) | same | *(ran)* on GPT-2, DBRX, MPT, BLOOM. |
| `vocab_size` (`vocab_size/n_vocab`) | `vocab_size` = `config.vocab_size` (`:383-385`) | same | |
| `head_dim` (`v_head_dim` → `head_dim` → `hidden//heads`, `ru.py:353-363`) | `head_dim` (`:396-399`) + `fam/deepseek_v2.py:44-46` override | same | Same result; DeepSeek rule is in the family, not the root. |
| `qk_head_dim` (`ru.py:366-372`) | `qk_head_dim` (`:401-404`) + `deepseek_v2.py:49-51` | same | |
| `num_kv_heads` (`multi_query`→1; `num_key_value_heads/num_kv_heads/n_head_kv`; `ru.py:375-385`) | `num_kv_heads` (`:391-394`) + `fam/falcon.py:157-162` | same | Neither reads MPT/DBRX `attn_config.kv_n_heads`; both return `num_heads` there. Not verified against those modules' real head counts. |
| `intermediate_size` (`n_inner` first, then `intermediate_size/ffn_hidden_size/ffn_dim`, DBRX `ffn_config`, MPT `expansion_ratio`, BLOOM 4×; `ru.py:388-409`) | `intermediate_size` = `config.intermediate_size` (`:406-409`) + family functions in `gpt2/gptj/falcon/opt/mpt/bloom.py` | changed | **DBRX has no family function and no `config.intermediate_size`: `model.intermediate_size` raises `AttributeError`** *(ran on `yujiepan/dbrx-tiny256-random`)*. `families.md:96` documents the width but no code publishes it. |
| `linear_num_value_heads`, `linear_key_head_dim`, `linear_value_head_dim` (`st.py:245-248`) | none; the suite reads `module.num_v_heads/head_k_dim/head_v_dim` (`suite.py:455`) | missing | |
| `linear_attention_kernels: dict` (`st.py:252`, `ru.py:1047-1099`) | none; `route_kernels` / `route_delta_rule` rebind module globals instead (`comp/recurrent.py:95-143`) | changed | See §3.9. |
| `block_structure: str` (`ru.py:1515-1532`) | none | missing | `contributing.md:217-218` marks trivial. |
| `is_vllm: bool` | none | missing | No vLLM path. |
| `remote: bool` | none | missing | |
| `internals: Internals` (`internals.py:14`) | `model.support()` (`S.py:287-313`), `Standard.values()` (`comp/standard.py:62-65`) | changed | No forward-order `rank` (`internals.py:59-70`); `contributing.md:181-184` open. |
| — | `model.family: ModuleType` (`S.py:122`, `:142`) | new | |
| — | `model.embed_tokens`, `model.norm`, `model.lm_head` typed attrs (`S.py:124-126`) | renamed | nnterp 1.x: `model.embed_tokens`, `model.ln_final`, `model.lm_head`; nnterp 1.x containers stay under `model.model` (`ru.py:159`), nnterp lifts them to the root (`README.md:23-24`). |

### 1d. Accessors (per-layer rows of nnterp 1.x's address table, `ru.py:1198-1291`, `:1319-1373`, `:1422-1512`)

nnterp 1.x spelling is `model.<row>[i]` (read) / `model.<row>[i] = v` (write); whole-model rows are `model.<row>()` / `model.<row>[None] = v` (`ru.py:829-864`).

| nnterp 1.x | nnterp equivalent | status | note |
|---|---|---|---|
| `embeddings_input()` (`:1203`) | `model.embed_tokens.input` (plain nnsight) | renamed | |
| `embeddings_output()` (`:1204`) / `token_embeddings` | `model.token_embeddings` (`S.py:186-194`) or `embed_tokens.output` | same | |
| `ln_final_output()` (`:1205`) | `model.norm.output` | renamed | |
| `lm_head_output()` (`:1206`) | `model.lm_head.output` | renamed | |
| `logits()` row (`:1210`) / `model.logits` (`st.py:738-743`) | `model.logits` (`S.py:170-184`) | same | Both read `output.logits` (softcapped on Gemma-2); nnterp is assignable. |
| `layers_input[i]` (`:1211`) | `model.layers[i].input` | renamed | |
| `attentions[i]` (`:1212`) | `model.layers[i].self_attn` | renamed | On DBRX nnterp 1.x's `self_attn` is `norm_attn_norm` (the container, `ru.py:203`); nnterp's is `norm_attn_norm.attn` (`fam/dbrx.py:24`), so `attentions_input[i]` (pre-norm block input) ≠ nnterp `self_attn.input` (normed) there. |
| `attentions_input[i]` (`:1213`) | `model.layers[i].self_attn.input` | renamed | DBRX caveat above. |
| `attentions_norm_output[i]` (`:1320-1331`) | none; `self_attn.input` is the same tensor on pre-norm blocks (`README.md:32`, `suite.py:222-245`) | missing | No norm-output value; norm names are deliberately not standardized (`README.md:40-47`). |
| `attentions_premix[i]` = `o_proj`-family input (`:1332-1335`) | none | missing | `contributing.md:213-215` open. `attention_head_outputs` reshaped is the same numbers. |
| `attention_queries[i]` (`:1219-1222`, `Index(0,1)` of interface `inputs`) | `layers[i].self_attn.attention_queries` (`comp/attention.py:87-97`) | renamed | nnterp also covers GPT-J/BLOOM/MPT/Falcon (§1i); nnterp 1.x marks those `_no_interface` (`ru.py:1393-1404`). |
| `attention_keys[i]` (`:1223-1226`) | `.attention_keys` (`:99-107`) | renamed | |
| — (nnterp 1.x has no values row; `INTERFACE_ROWS` `ru.py:1001`) | `.attention_values` (`:109-116`) | new | |
| `attention_scores[i]` (`:1230-1233`; GPT-OSS: `F_softmax_0` input, one key wider, tag `sink`, `:1507-1510`) | `.attention_scores` (`:118-125`; GPT-OSS: `attn_weights_1`, same width as the pattern, `fam/gpt_oss.py:40-42`) | changed | On GPT-OSS the two read different tensors; see §3.14. |
| `attention_probabilities[i]` (`:1237-1242`, dropout output; `disable`d unless `enable_attention_probs=True`, `st.py:292-295`) | `.attention_probabilities` (`:143-160`, dropout output; unavailable unless eager) | changed | Same op; gating differs (§3.2). |
| `attention_head_outputs[i]` (`:1246-1248`, `Index(0)` of interface output, `[b,seq,heads,hd]`) | `.attention_head_outputs` (`:162-172`) | renamed | |
| `attentions_output[i]` (`:1252`; families `:1435`, `:1442`, `:1476`) | `.attention_output` (`:127-141`; `fam/gemma2.py:30-35`, `bloom.py:73-79`) | renamed | Same contribution semantics on Gemma-2/3, OLMo-2, BLOOM, DBRX. nnterp 1.x BLOOM reads `self_attn.dense` output; nnterp reads `dropout_add_0` input (works under `slow_but_exact` too, *(ran)*). |
| `mlps[i]` (`:1253`) | `model.layers[i].mlp` | renamed | |
| `mlps_input[i]` (`:1254`) | `model.layers[i].mlp.input` | renamed | |
| `mlps_norm_output[i]` (`:1349-1362`) | none (`mlp.input`) | missing | as `attentions_norm_output`. |
| `layers_mid[i]` (`:1336-1348`) | none; `layers[i].input + self_attn.attention_output` | missing | `contributing.md:213-215`. |
| `mlps_activation[i]` (`:1363-1366`) | none | missing | |
| `mlps_neurons[i]` (`:1367-1372`) | none | missing | |
| `mlps_output[i]` (`:1255`; `:1436`, `:1443`, `:1460`) | `.mlp_output` (`comp/mlp.py:17-31`; `fam/gemma2.py:41-46`, `bloom.py:92-98`, `mpt.py:84-89`, `falcon.py:136-148`) | renamed | Falcon: nnterp reads a clone (§3.15); nnterp 1.x reads the live tensor. |
| `layers_output[i]` (`:1256`, `FirstIfTuple`) | `layers[i].layer_output` (`comp/layer.py:58-77`, `first_tensor`/`rewrap`) | renamed | Same unwrap/rewrap semantics. |
| `linear_attentions_state_input[i]` (`:1261-1263`, `Copied()` at mixer's `recurrent_state_0`) | `layers[i].linear_attn.state_input` (`comp/linear_attention.py:76-85`, kernel `initial_state` kwarg, cloned) | renamed | Different op, same meaning (None on a fresh prompt). |
| `linear_attention_queries[i]` (`:1264`) | `linear_attn.attention_queries` (`:172-175`) | renamed | |
| `linear_attention_keys[i]` (`:1267`) | `linear_attn.attention_keys` (`:177`) | renamed | |
| `linear_attention_values[i]` (`:1270`) | `linear_attn.attention_values` (`:182`) | renamed | |
| `linear_attention_decays[i]` (`:1273`) | `linear_attn.decays` (`:187`) | renamed | |
| `linear_attention_betas[i]` (`:1276`) | `linear_attn.betas` (`:192`) | renamed | |
| `linear_attention_head_outputs[i]` (`:1279`) | `linear_attn.attention_head_outputs` (`:208`) | renamed | |
| `linear_attentions_state_output[i]` (`:1282-1289`, input of `cache_params_update_recurrent_state_0`, with `.scan`) | `linear_attn.state_output` (`:213-216`, kernel return 1) | renamed | `.scan(layer, cuts, edit)` → `state`/`states`/`state_after`/`set_state_after` (§3.9). |
| `linear_attentions_output[i]` (`:1290`) | `linear_attn.attention_output` (`:163-170`) | renamed | |
| — | `linear_attn.state`, `.states`, `.state_after(t)`, `.set_state_after(t, v)` (`comp/recurrent.py:352-446`) | new | Per-token state via `tracer.iter`; needs `route_kernels(family, "torch")`. |

### 1e. Properties and methods on the model

| nnterp 1.x (`st.py`) | nnterp (`S.py`) | status | note |
|---|---|---|---|
| `attn_probs_available` (`:370-375`) | `model.support()["self_attn.attention_probabilities"] is None` (`:260-286`) | changed | Bool → reason dict. |
| `input_ids` (`:377-384`, read-only property) | `input_ids` (`:328-336`, assignable EProperty) | same + | nnterp can assign: the model then runs on the new ids. |
| `input_size` (`:386-393`) | `input_size` (`:348-355`) | same | Both `torch.Size`; nnterp refuses assignment with `AttributeError`. |
| `attention_mask` (`:395-402`) | `attention_mask` (`:338-346`, assignable) | same + | |
| `token_embeddings` get/set (`:404-413`) | `token_embeddings` (`:167-175`) | same | |
| `next_token_probs` (`:415-419`) | `next_token_probs` (`:177-192`) | same | Both read-only in practice; nnterp names the reason on assignment. |
| `skip_layer(layer, skip_with)` (`:421-429`) | none: `skip_layers(i, i)` or `layers[i].skip_with(h)` (`comp/layer.py:47-56`) | missing | |
| `skip_layers(start_layer, end_layer, skip_with, layer_returns_tuple)` (`:431-464`; tuple filler `(t, DummyCache())`) | `skip_layers(start, end, skip_with)` (`:196-210`; filler `(hidden, None)`; negative indices) | changed | `layer_returns_tuple` gone; `Layer.returns_tuple` is a family constant (`comp/layer.py:45`). |
| `detect_layer_output_type()` (`:330-353`) | none | dropped | Not needed: tuple-ness is declared per family. |
| `steer(layers, steering_vector, factor, positions*, token_positions, batch_index)` (`:466-535`; sorts `layers`; vLLM clone path) | `steer(layers, vector, factor, token_positions, batch_index)` (`:212-231`; no sort; in-place only) | changed | `positions` removed; second positional renamed `vector`; caller must pass `layers` ascending. List `batch_index` works *(ran)* despite the `int \| None` annotation. |
| `project_on_vocab(hidden_state)` = `lm_head(ln_final(h))` (`:537-551`) | `project_on_vocab(hidden)` = `lm_head(norm(h))` **then `final_logit_softcapping`** (`:233-242`) | changed | Differs on Gemma-2/3 only; nnterp's equals `logits` at the last block (`suite.py:435-439`). |
| `probs_to_dict(tokens, probs)` (`:553-560`, `convert_ids_to_tokens`) | `probs_to_dict(probs, k=5)` (`:244-247`, `tokenizer.decode`) | changed | Different signature and different token strings (`decode` gives `" Paris"`, `convert_ids_to_tokens` gives `"ĠParis"`). |
| `get_topk_closest_tokens(hidden_state, k=5)` → dict for 1-D, list for 2-D; checks `hidden_size` (`:562-595`) | `get_topk_closest_tokens(hidden, k=5)` → always `list[dict]`, row-major over leading axes, no shape check (`:249-256`) | changed | 1-D input returns `[{...}]` *(ran)*. |
| `add_prefix_false_tokenizer` (`:355-368`) | `add_prefix_false_tokenizer` (`:359-369`) | same | |
| `logits` property (`:738-743`) | `logits` EProperty (`:151-165`) | same | |
| `dispatch()` re-pins kernels (`:732-736`) | inherited nnsight `dispatch()` | changed | nnterp needs no re-pin: `route_delta_rule` rebinds module globals (§3.9). |
| `_remoteable_class()` → `TransformersModel` (`:729-730`) | same (`:413-422`) | same | |
| — | `support(layer=None)` (`:260-286`) | new | |
| — | `family`, sizes as `StandardizedProperty` (`:25-49`) | new | |

### 1f. `StandardizedVLM` (`st.py:746-824`) and `StandardizedVLLM` (`standardized_vllm.py`)

| nnterp 1.x | nnterp | status | note |
|---|---|---|---|
| `StandardizedVLM(model, check_renaming, remote, allow_dispatch, enable_attention_probs, check_attn_probs_with_trace, allow_multimodal, rename_config, tokenizer_kwargs, **kw)`; `task="image-text-to-text"`; `trace(prompt, images=...)` | none | missing | `S.py:142` does read `text_config.model_type`, so a multimodal config resolves its text family, but `task` is text-generation and no `language_model` lift exists (`contributing.md:194-196`). |
| `allow_multimodal` (heterogeneous layer classes; `ru.py:1902-1912`) | none | missing | nnterp's `ENVOYS` is class-keyed; a cross-attention block would be a plain `Envoy` with no `layer_output` (`contributing.md:185-189`). |
| `StandardizedVLLM(...)` (`standardized_vllm.py:13-166`): experimental gate `:81-97`, `tensor_parallel_size` default `:98-101`, prefix-cache refusal `:102-112`, `trace()` `max_tokens=1` `:131-143`, `generate()` `:145-156`, HF→vLLM kwargs `:158-166`, `hf_kwargs_to_vllm_kwargs` (`ru.py:1946-1964`) | none | missing | Not listed in `contributing.md` at all. |

### 1g. `nnsight_utils.py`

| nnterp 1.x (`nnterp/nnsight_utils.py`) | nnterp (`nnterp/nnsight_utils.py`) | status | note |
|---|---|---|---|
| `GetModuleOutput = Callable[[TransformersModel, int], TraceTensor]` (`:16`) | `GetActivations = Callable[[StandardizedTransformer, int], Tensor]` (`:17`) | renamed | |
| `get_embed_tokens :19`, `get_layers :26`, `get_num_layers :35`, `get_layer :46`, `get_layer_input :58`, `get_layer_output :70`, `get_attention :83`, `get_attention_output :95`, `get_mlp :111`, `get_mlp_output :118`, `get_logits :127`, `get_unembed_norm :138`, `get_unembed :151`, `project_on_vocab :162`, `get_next_token_probs :175`, `set_layer_output :186` | none; `layer_output(model, layer)` (`:20-22`) is the only accessor function | missing | These accepted a raw `TransformersModel`; nnterp's helpers are typed on `StandardizedTransformer` only. |
| `ModuleAccessor` (`:203-256`) | none | missing | |
| `get_token_activations(nn_model, prompts=None, layers=None, get_activations=None, remote=False, idx=None, tracer=None)` (`:260-316`) → `[layers, prompts, hidden]` CPU | `get_token_activations(model, prompts=None, layers=None, get_activations=None, remote=False, idx=None, tracer=None)` (`:33-75`) → same shape | same | In-tracer branch: nnterp 1.x `.to(device of layer 0 output)` (`:313-315`); nnterp leaves the value where it is (`:73-74`). Padding-side checks identical (`:25-30`). |
| `collect_last_token_activations_session(nn_model, prompts, batch_size, layers, get_activations, remote, idx)` (`:320-378`) | same signature (`:78-101`) | same | |
| `collect_token_activations_batched(nn_model, prompts, batch_size, layers, get_activations, remote, idx, tqdm=None, use_session=True)` (`:381-431`) | same signature (`:104-128`) | same | |
| `compute_next_token_probs(nn_model, prompt, remote=False)` (`:434-449`) → `[prompts, vocab]` CPU | same (`:131-135`, via `model.next_token_probs`) | same | |

### 1h. `prompt_utils.py`

| nnterp 1.x (`nnterp/prompt_utils.py`) | nnterp (`nnterp/prompt_utils.py`) | status | note |
|---|---|---|---|
| `TokenizationError` (`:14`) | `TokenizationError` (`:21`) | same | |
| `get_first_tokens(words, llm_or_tokenizer, use_hacky_implementation=False) -> list[int]` (`:18-94`); accepts raw `TransformersModel`; falls back to `model.tokenizer` if `add_prefix_false_tokenizer` fails (`:39-46`) | `get_first_tokens(words, model_or_tokenizer, use_hacky_implementation=False) -> list[int]` (`:25-69`); `StandardizedTransformer` or tokenizer only, no fallback | same | Same return (first tokens of `word` and `" word"`, deduplicated); raw-model input dropped. |
| `Prompt(prompt, target_tokens, target_strings=None)` (`:97-110`) | same dataclass (`:72-84`) | same | |
| `Prompt.from_strings(prompt, target_strings, tokenizer)` (`:112-129`) | `Prompt.from_strings(prompt, target_strings, model_or_tokenizer)` (`:86-97`) | same | Third parameter renamed; positional use unchanged. |
| `Prompt.has_no_collisions(ignore_targets=None) -> bool` (`:131-142`) | same (`:99-103`) | same | |
| `Prompt.get_target_probs(probs, layer=None) -> dict[str, Tensor]` (`:144-153`), `probs` `[batch, layers, vocab]` | same (`:105-110`) | same | |
| `Prompt.run(nn_model, get_probs)` (`:155-161`) | `run(model, get_probs)` (`:112-115`) | same | |
| `next_token_probs_unsqueeze(nn_model, prompt, remote=False, **_)` → `[batch, 1, vocab]` (`:164-168`) | same (`:118-120`) | same | |
| `run_prompts(nn_model, prompts, batch_size=32, get_probs_func=None, func_kwargs=None, remote=False, tqdm=tqdm)` → `{target: [prompts, layers]}` (`:171-224`) | same, but `tqdm=None` default (`:123-156`) | changed | Progress bar off by default in nnterp. |

### 1i. `rename_utils.py` (public names) and `internals.py`

| nnterp 1.x | nnterp | status | note |
|---|---|---|---|
| `RenamingError` (`ru.py:34`) | `Unavailable` (`comp/eproperty.py:21`, `RuntimeError`), `UnsupportedFamily` (`fam/__init__.py:35`, `ValueError`), nnsight `SourceNotAvailable` for a moved op | changed | Three typed errors replace one. |
| `AttnProbFunction` (`:38-50`) / `RenameConfig.attn_prob_source` | `EProperty(key=callable)`, a key function returning a path (`comp/eproperty.py:86-96`, `:154-156`), e.g. `fam/falcon.py:51-58`, `comp/recurrent.py:247-254`, `:311-329` | changed | |
| `RenameConfig(attn_name, mlp_name, ln_final_name, lm_head_name, model_name, layers_name, ...)` (`:142-147`) | `rename=` kwarg / `family.RENAME` | changed | |
| `RenameConfig.ignore_mlp / ignore_attn` (`:149-150`) | none; a family simply omits the `Mlp` key (`fam/opt.py:44`) | changed | |
| `RenameConfig.attn_head_config_key / hidden_size_config_key / vocab_size_config_key` (`:151-153`) | `def <size>(model)` in the family module (`S.py:26-50`; `fam/gpt2.py:63-65`) | changed | Via `families.register()` for a user family. |
| `RenameConfig.attn_output_source / mlp_output_source` (`:154-155`) | override `attention_output` / `mlp_output` on an `Attention`/`Mlp` subclass, passed via `envoys=` (`docs/extending/overriding-values.md`) | changed | |
| `RenameConfig.addresses: dict[str, Address]` (`:156`) | a new `EProperty` on an envoy subclass via `envoys=` (`test_registry.py:106-118`) | changed | |
| `MODEL_NAMES`, `ATTENTION_NAMES`, `LINEAR_ATTENTION_NAME`, `LAYER_NAMES`, `LN_NAMES`, `LM_HEAD_NAMES`, `MLP_NAMES`, `EMBED_TOKENS_NAMES` (`:159-236`) | per-family `RENAME`; their union is `nnterp.families.default`'s `RENAME` | changed | The best-effort default family, which checks itself at load (`docs/usage/loading.md`). |
| `bloom_slow_but_exact` (`:187-195`) + `_NO_CONTRIBUTION` disable (`:1453-1458`) | none needed: `dropout_add_0` input is read on that path too *(ran, identity holds)* | changed (better) | |
| `text_config(model)` (`:272-276`) | `_read_config` picks `text_config.model_type` (`S.py:142`); sizes read `model.config` directly | changed | On a VLM config nnterp's `hidden_size` would read the top-level config; untested since nnterp does not load VLMs. |
| `get_num_attention_heads/get_hidden_size/get_vocab_size/get_head_dim/get_qk_head_dim/get_num_kv_heads/get_intermediate_size(model)` (`:279-409`) | the `StandardizedProperty` rows (`S.py:399-433`) | renamed | Free functions on a raw model are gone. |
| `IOType {INPUT, INPUTS, OUTPUT}` (`:412-425`) | the last segment of an `EProperty` path, `input`\|`inputs`\|`output` (`comp/eproperty.py:162-191`) | renamed | |
| `get_attention_layers(layers)` (`:428-439`), `linear_attention_error` (`:442-447`) | none | missing | |
| `Selection`, `Index(*steps)`, `Copied()`, `FirstIfTuple()` (`:450-513`) | `select: int \| str` (one step only, `comp/eproperty.py:195-223`); `first_tensor`/`rewrap` (`comp/standard.py:13-21`); a `preprocess` that clones (`linear_attention.py:76-85`) | changed | No multi-step `Index(0, 1)`; nnterp uses `"source.attention_interface_1.inputs", select=1`. |
| `Address(module, io, op, select, order, unavailable, tags, per_layer, seq_axis, width, heads, keys, needs, scan)` (`:516-631`) | `EProperty(key, description, unavailable, select)` with a path for a key (`"output"`, `"../norm.output"`, `"source.<op>.inputs"`), `DerivedEProperty(compute, ...)` (`comp/eproperty.py`) | changed | Lost: `order`, `tags`, `seq_axis`, `width`, `heads`, `keys`, `needs`, `scan`. Gained: `.layout`/`.dims` (`:122-145`), `description` in the repr. |
| `LayerAccessor.unavailable_on(layer)` (`:700-723`) | `EProperty.reason(envoy)` (`:103-105`), `Standard.support()` | renamed | |
| `LayerAccessor.num_heads`, `.width` (`:734-752`) | none; `.dims` names axes but not sizes | missing | |
| `LayerAccessor.scan(layer, cuts, edit)` (`:754-778`) | `LinearAttention.states/state_after/set_state_after` (§3.9) | changed | |
| `LayerAccessor.disable(reason)` (`:780-786`) | none; `unavailable=` is declared on the class | missing | No runtime disabling. |
| `LayerAccessor.get_module(layer)` (`:788-795`) | `envoy.get(path)` (nnsight) | renamed | |
| `LayerAccessor.get_operation(layer, containing_source)` (`:797-820`) | `EProperty._resolve` (private, `comp/eproperty.py:162-191`) | missing | No public op accessor; error lists what `.source` has. |
| `LayerAccessor.returns_tuple(layer)` (`:866-873`) | `Layer.returns_tuple` class attr (`comp/layer.py:45`) | changed | Declared, not detected. |
| `LayerAccessor.print_source(layer)` (`:875-902`) | `print(envoy.source)` (`docs/extending/finding-source-ops.md`) | renamed | |
| `check_attention_probabilities(model, layer, allow_dispatch, use_trace)` (`:905-986`) | `suite.py:275-289`, `:303-317` (tests only) | dropped | |
| `_INTERFACE`/`INTERFACE_ROWS` (`:996-1001`) | `INTERFACE` (`comp/attention.py:28`); `INTERIOR` in `suite.py:26` | renamed | |
| `delta_rule_call(mixer)` (`:1003-1044`, step-0 = chunked) | `RecurrentMixer.KERNEL` (`recurrent.py:311-329`) | changed | nnterp applies the forward's own test (`use_precomputed_states and seq_len == 1`) to the bindings the forward makes, so a warm-cache single-token trace resolves and several tokens over a cache read the prompt's kernel; nnterp 1.x documents it as unsupported (`:1023-1029`). |
| `pin_linear_attention_kernels(model)` (`:1047-1099`) | `route_kernels(family, "torch"\|"default")` / `route_delta_rule` (`recurrent.py:95-143`) + `needs_torch_kernels` (`:146-160`) | changed | §3.9. |
| `delta_rule_scan(accessor, layer, cuts, edit)` (`:1102-1162`) | `states`, `state_after`, `set_state_after` | changed | §3.9. |
| `DEFAULT_ADDRESSES` (`:1198-1291`) | `comp/layer.py`, `attention.py`, `mlp.py`, `linear_attention.py` base classes | renamed | |
| `BlockStructure`, `get_block_structure` (`:1293`, `:1515-1532`) | none | missing | |
| `STRUCTURAL_ADDRESSES`, `structural_addresses` (`:1319-1373`, `:1535-1584`) | none | missing | |
| `POST_SUBLAYER_NORM_MODEL_TYPES`, `post_sublayer_norm` (`:1410-1414`) | `EProperty("../post_attention_layernorm.output")` overrides in `fam/gemma2.py`, `gemma3_text.py`, `olmo2.py`, `olmo3.py` | renamed | |
| `FAMILY_ADDRESSES` (`:1422-1512`), `addresses_for` (`:1587-1606`) | `fam/*.py` + `families.lookup` | renamed | |
| `get_ignores`, `check_io`, `_check_has_module`, `_check_attention_layers`, `_warn_heterogeneous_types`, `_check_output_source`, `check_model_renaming` (`:1609-1943`) | none (suite) | dropped | `contributing.md:174-175`, `:185-189`. |
| `HF_TO_VLLM_KWARGS_MAP`, `hf_kwargs_to_vllm_kwargs` (`:1946-1964`) | none | missing | |
| `Internals[name]` (`internals.py:34-40`) | `type(envoy).<name>` / `Standard.values()[name]` | changed | |
| `Internals.status(layer=None)` (`:42-57`): a per-layer row is available if **any** layer has it | `model.support(layer=None)` (`S.py:287-313`): `None` only if available on **every** block, else `{layer: reason}` | changed | Opposite aggregation; keys are dotted (`"self_attn.attention_probabilities"`). |
| `Internals.rank(name, layer)` (`:59-70`) | none | missing | `contributing.md:181-184` open. |

### 1j. `interventions.py`, `display.py`, `utils.py`, `logging.py`, `__main__.py`

| nnterp 1.x | nnterp | status | note |
|---|---|---|---|
| `logit_lens(nn_model, prompts, remote=False, return_inv_logits=False)` → `[prompts, layers, vocab]` (`interventions.py:29-67`) | none; recipe in `docs/patterns/logit-lens.md` | missing | `contributing.md:190-193` open. |
| `TargetPrompt(prompt, index_to_patch)` (`:71-73`) | none | missing | |
| `repeat_prompt(words, rel, sep, placeholder, index_to_patch)` (`:76-99`) | none | missing | |
| `it_repeat_prompt(tokenizer, words, rel, sep, placeholder, complete_prompt, add_user_instr, use_system_prompt)` (`:102-165`) | none | missing | |
| `TargetPromptBatch` + `from_target_prompts/from_target_prompt/from_prompts/auto` (`:169-230`) | none | missing | |
| `patchscope_lens(nn_model, source_prompts, target_patch_prompts, layers, latents, remote)` → `[prompts, layers, vocab]` (`:234-300`) | none; recipe in `docs/patterns/activation-patching.md` | missing | |
| `patchscope_generate(nn_model, prompts, target_patch_prompt, max_length, layers, remote, max_batch_size)` → `{layer: tokens}` (`:304-355`) | none | missing | |
| `patch_object_attn_lens(nn_model, source_prompts, target_prompts, attn_idx_patch, num_patches)` (`:358-405`) | none | missing | |
| `steer` — not in `interventions.__all__` (`:17-25`); it is the model method | `model.steer` | same | |
| `plot_topk_tokens(next_token_probs, tokenizer, k, title, use_token_ids, file, save_html, height, width) -> go.Figure` (`display.py:13-116`) | none | missing | `contributing.md:192`. |
| `prompts_to_df(prompts, tokenizer=None) -> DataFrame` (`:119-131`) | none | missing | |
| `TraceTensor` (`utils.py:14`) | none (plain `torch.Tensor` annotations) | dropped | |
| `ArchitectureNotFound` + 14 guarded class imports (`:22-87`) | family modules import their own classes (`fam/*.py`) | dropped | |
| `detect_automodel` (`:90-179`) | none | missing | |
| `is_notebook`, `display_markdown`, `display_source` (`:182-201`) | none | missing | |
| `DummyCache` (`:204-206`) | `None` in the tuple (`comp/layer.py:56`) | changed | |
| `dummy_inputs` (`:209-213`), `try_with_scan(...)` (`:216-273`) | none | dropped | |
| `unpack_tuple` (`:276-279`) | `first_tensor` (`comp/standard.py:13-15`) | renamed | |
| `logging.logger` (`logging.py`) | none | dropped | nnterp emits no log lines; warnings via `warnings.warn` (`prompt_utils.py:63`). |
| `python -m nnterp run_tests --model-names/--class-names` (`__main__.py`) | `HF_HUB_OFFLINE=1 pytest` | dropped | |

## 2. Missing in nnterp

**Accessors / values**
- `attentions_norm_output`, `mlps_norm_output` (`ru.py:1320-1331`, `:1349-1362`): a norm-output value keyed by block structure. Porting: two `EProperty`s with a child path on `Layer` per family (the norm name differs per family, so a family constant). nnterp's docs argue `self_attn.input`/`mlp.input` are the same tensor and refuse to standardize norm names (`README.md:40-47`); not listed as open.
- `layers_mid` (`:1336-1348`): the mid-block stream. Porting: an `EProperty("post_attention_layernorm.input")`-style value with a per-structure name and an `unavailable` on parallel blocks. Open in `contributing.md:213-215`.
- `attentions_premix` (`:1332-1335`): `o_proj`-family input. Porting: `EProperty("o_proj.input")` with the projection name per family. Open (`:213-215`).
- `mlps_activation`, `mlps_neurons` (`:1363-1372`): act-fn output and down-projection input, with a per-layer MoE `unavailable` (`:1553-1574`). Porting: two `EProperty`s with a child path on `Mlp` plus a MoE predicate. Open (`:213-215`).
- `attention_layers` / `linear_attention_layers` (`st.py:224-226`), `block_structure` (`:189`): Porting: three `StandardizedProperty`s. Open, marked trivial (`contributing.md:216-218`).
- `linear_num_value_heads`, `linear_key_head_dim`, `linear_value_head_dim` (`st.py:245-248`): Porting: three `StandardizedProperty`s reading `text_config`. Not listed.
- `LayerAccessor.width` / `.num_heads` (`ru.py:734-752`): expected sizes per value. Porting: a `sizes(model)` on `EProperty` mapping `.dims` to model sizes (the suite already does this by hand, `suite.py:447-456`). Listed indirectly under "Layout facts" (`contributing.md:208-212`).
- `Internals.rank` / `Address.order` (`internals.py:59-70`): Porting: an `order` int on each descriptor plus a `model.rank(...)`. Open (`contributing.md:181-184`).
- `LayerAccessor.disable(reason)` (`ru.py:780-786`): runtime disabling. Porting: an instance-level `unavailable` override. Not listed.
- `Internals.status` "available on any layer" aggregation (`internals.py:55-56`): nnterp's is "on every block". Not a port target, but a migration trap (§3.4).

**Model surface**
- `skip_layer(layer, skip_with)` (`st.py:421-429`): one-liner over `skip_layers`. Not listed.
- `steer(..., positions=)` deprecated alias (`st.py:471`, `:486-496`), vLLM clone path (`:505-524`), and the ascending sort (`:500`). Not listed.
- `get_topk_closest_tokens` shape check and dict-for-1-D return (`st.py:577-586`). Not listed.
- `probs_to_dict(tokens, probs)` (`st.py:553-560`): nnterp's is `(probs, k)`. Not listed.
- `attn_probs_available` bool (`st.py:370-375`). Not listed; `support()` covers it.
- `detect_layer_output_type()` (`st.py:330-353`). Deliberately unnecessary.
- `dispatch()` re-pin (`st.py:732-736`). Unnecessary under `route_delta_rule`.
- `intermediate_size` on DBRX (`ru.py:399-400`): **a bug, not a design choice** — `fam/dbrx.py` has no `intermediate_size(model)` and `DbrxConfig` has no `intermediate_size`, so the root property raises *(ran)*. Porting: `def intermediate_size(model): return model.config.ffn_config.ffn_hidden_size` in `fam/dbrx.py`. Not listed.

**Loading / validation**
- `check_renaming`, `check_model_renaming`, `check_io`, `_check_attention_layers`, `_check_output_source`, heterogeneous-layer refusal, `try_with_scan` (`ru.py:1609-1943`, `utils.py:216-273`): Porting: a `model.validate()` running the suite's identity and shape checks on a dummy input. Listed as deliberate ("nothing runs at construction", `contributing.md:174-175`) and as open for the causal/heterogeneous/`residual`-argument checks (`:185-189`).
- `check_attention_probabilities` causal write (`ru.py:905-986`). Open (`:185-189`).
- `enable_attention_probs=True` forcing eager (`st.py:318-324`). Deliberate (`:171-173`).
- `allow_dispatch`, `check_attn_probs_with_trace`, `remote=` construction semantics (`st.py:168-176`): nnterp has no scan-based checks to gate. Deliberate/open ("Real NDIF verification", `:197-199`).
- `device_map="auto"` default (`st.py:316`). Open, trivial (`:216`).
- `RenameConfig` as a user-side extension without a registry entry (`ru.py:53-156`), the global name lists (`:159-236`), Deliberate. An unknown model loads with no code through the default family, with a warning (`tests/families/test_default.py`).
- `bloom_slow_but_exact` handling (`ru.py:187-195`): unnecessary in nnterp (works). Not listed.

**Interventions**: `logit_lens`, `TargetPrompt`, `repeat_prompt`, `it_repeat_prompt`, `TargetPromptBatch`, `patchscope_lens`, `patchscope_generate`, `patch_object_attn_lens` (`interventions.py:29-405`). Porting: mechanical, over `model.layers[i].layer_output`, `model.project_on_vocab`, `model.next_token_probs`; `patch_object_attn_lens` needs `layers[i].self_attn.input`. Open (`contributing.md:190-193`; `it_repeat_prompt` is not named there).

**Display**: `plot_topk_tokens`, `prompts_to_df` (`display.py:13-131`). Porting: copy as an optional extra. Open (`:192`).

**VLM / vLLM**
- `StandardizedVLM`, `detect_automodel`, `text_only`, `load_model`, `allow_multimodal` (`st.py:746-824`, `utils.py:90-179`, `__init__.py:21-66`). Porting: a `task=` switch plus a `language_model.*` container lift in each family's `RENAME` and a class-keyed envoy for cross-attention blocks. Open (`contributing.md:194-196`).
- `StandardizedVLLM` and the vLLM row surgery (`standardized_vllm.py`, `st.py:193-203`, `ru.py:1946-1964`). Porting: a `VLLM` subclass with the same family lookup; every `source.`-keyed value would be unavailable (no interface, logits outside the forward). Not listed in `contributing.md`.

**Helpers**
- `nnsight_utils`: the sixteen raw-model accessors and `ModuleAccessor` (`nnsight_utils.py:19-256`). Porting: trivial but they exist to serve raw `TransformersModel`s, which nnterp does not target. Not listed.
- `prompt_utils.get_first_tokens` on a raw `TransformersModel` and the `model.tokenizer` fallback (`prompt_utils.py:39-48`). Not listed.
- `run_prompts` progress bar default (`:179`). Not listed.
- `utils.is_notebook`, `display_markdown`, `display_source`, `TraceTensor`, `DummyCache`, `dummy_inputs`. Not listed.
- `logging.logger` and every `logger.info/warning` at load (hybrid notice `ru.py:1812-1817`, kernel pinning `st.py:254-259`, ignores `ru.py:1635`). nnterp is silent at load. Not listed.
- `__main__.run_tests`, `data/*.json` status tracking, cross-transformers-version scripts, `conftest --model-names/--class-names/--save-test-logs`. Not listed.

## 3. Different in nnterp (breaking for an nnterp 1.x user)

1. **Accessor spelling and home.** nnterp 1.x: `model.layers_output[i]`, a `LayerAccessor` attribute of the model (`st.py:209-218`); whole-model rows `model.ln_final_output()`. nnterp: `model.layers[i].layer_output`, an `EProperty` on the block envoy (`comp/layer.py:58`), `model.norm.output`. Migration: `layers_output[i]`→`layers[i].layer_output`; `attentions_output[i]`→`layers[i].self_attn.attention_output`; `mlps_output[i]`→`layers[i].mlp.mlp_output`; `attention_probabilities[i]`→`layers[i].self_attn.attention_probabilities`; `layers_input[i]`→`layers[i].input`; `attentions_input[i]`→`layers[i].self_attn.input`; `ln_final`→`norm`; `model.model.layers`→`model.layers`.

2. **Attention pattern gating.** nnterp 1.x: `enable_attention_probs=True` forces eager and runs the causal check; without the flag the row is `disable`d even if you passed `attn_implementation="eager"` yourself (`st.py:277-295`, `:318-324`). nnterp: the constructor forces nothing; `needs_eager` (`comp/attention.py:18-23`) reports `sdpa` as the reason, and BLOOM/MPT need no eager at all (`fam/bloom.py:8-9`, `mpt.py:6-8`). Migration: replace `enable_attention_probs=True` with `attn_implementation="eager"`; drop `check_attn_probs_with_trace`.

3. **No load-time validation.** nnterp 1.x runs `check_model_renaming` + `check_io` (and the pattern write) on a dummy input at construction (`st.py:264-282`, `ru.py:1892-1943`), so a mis-renamed model fails at load with the fixing argument. nnterp runs nothing for a shipped family (`S.py:128-155`); an unknown `model_type` gets the default family, which checks its names and a shape-only scan at load and raises `UnsupportedFamily` when they fail (`fam/default.py`), but a wrong op name surfaces as `SourceNotAvailable` inside the first trace (`comp/eproperty.py:186-190`). Migration: run `model.support()` and one trace of the values you need before an experiment.

4. **`support()` shape and aggregation.** nnterp 1.x `model.internals.status()` keys are row names and a per-layer row is `None` if *any* layer has it (`internals.py:42-57`). nnterp `model.support()` keys are dotted (`"self_attn.attention_probabilities"`), a value is `None` only if available on *every* block, else `{layer: reason}`; `support(layer=i)` is flat (`S.py:287-313`). Migration: `internals.status()["attention_probabilities"] is None` → `model.support()["self_attn.attention_probabilities"] is None`, and expect a dict on hybrids.

5. **`hasattr` raises.** nnterp 1.x: an accessor is a plain attribute; `hasattr(model, "attention_probabilities")` is `True` and the read raises `RenamingError` (`ru.py:788-791`). nnterp: `hasattr(envoy, "attention_probabilities")` raises `Unavailable` *(ran)* (`comp/eproperty.py:28-32`, `:122-123`; `test_base.py:51-52`). Migration: ask `support()`; never `hasattr`/`getattr(..., None)` on a value.

6. **`attn_probs_available` is gone.** `st.py:370-375` → `model.support()["self_attn.attention_probabilities"] is None`.

7. **Remote contract.** nnterp 1.x: `remote=True` at construction sets `allow_dispatch=False`, runs the checks with `scan()` on meta, ships nothing, server needs nnterp 1.x at the same version (`st.py:168-176`, CHANGELOG `:160-178`). nnterp: `remote=True` on `trace()`; the constructor kwarg is accepted only because nnsight accepts it *(ran)*; the block re-runs against the client's envoy tree, families by reference, server needs nnterp at the same version (`S.py:437-446`, `docs/usage/remote.md`). Neither has been exercised against a live NDIF (`contributing.md:197-199`). Migration: drop the constructor `remote=` (harmless) and keep it on `trace()`.

8. **Sizes.** Same values on every family checked except DBRX `intermediate_size` (raises in nnterp, §2). nnterp sizes are read-only descriptors (`S.py:49-50`); nnterp 1.x's are plain attributes you could overwrite. nnterp 1.x returns `None` where a config key is missing (`st.py:238-242`); nnterp raises `AttributeError`. Migration: none, except guard DBRX.

9. **DeltaNet API.** nnterp 1.x: rows `linear_attention_*[i]` / `linear_attentions_state_{input,output}[i]` on the model; kernels pinned to the PyTorch references at load (`ru.py:1047-1099`), so `fla`/`causal_conv1d` installed still works; the kernel op chosen by nnsight's step (`:1003-1044`); per-position state via `accessor.scan(layer, cuts, edit)` re-running the chunked kernel in pieces (`:1102-1162`); `linear_attention_kernels` dict; a warm-cache single-token trace unsupported. nnterp: values on `layers[i].linear_attn` (`comp/linear_attention.py:51-95`); kernel chosen by the forward's own test, the `use_precomputed_states_0` binding and the call's length (`comp/recurrent.py:311-329`); with `fla`/`causal_conv1d` installed every value but `attention_output` is *unavailable* (`needs_torch_kernels`, `recurrent.py:146-160`); per-token `state`/`states`/`state_after`/`set_state_after` exist only after `route_kernels(model.family, "torch")`, a process-wide rebinding of the module's kernel globals that must precede the first trace of a linear block (`recurrent.py:95-129`, `:352-446`). Migration: `linear_attentions_state_output.scan(i, cuts)` → `route_kernels(...)` then `mix.states`/`state_after(t)`; `pin_linear_attention_kernels` has no counterpart — uninstall `fla` or accept unavailability.

10. **`input_ids` / `input_size` / `attention_mask`.** Same reads (`self.inputs[1][...]`, `st.py:377-402`; `S.py:352-379`). nnterp's `input_ids` and `attention_mask` are assignable and re-run the model on the new tensors (`S.py:357-370`); nnterp 1.x's are read-only. Both serve at the model input: read before any block.

11. **`logits`.** Same: `output.logits`, softcap already applied on Gemma-2 (`st.py:738-743`, `ru.py:1210`; `S.py:170-184`). nnterp additionally assignable. `lm_head_output()` → `model.lm_head.output` for the raw projection.

12. **`token_embeddings`.** Same: `embed_tokens.output` (`st.py:404-413`; `S.py:186-194`).

13. **`project_on_vocab` applies softcap in nnterp** (`S.py:267-269`) and not in nnterp 1.x (`st.py:550-551`). On Gemma-2/3 a logit lens from nnterp 1.x and nnterp differ by `cap·tanh(x/cap)`. Migration: none for other families.

14. **`attention_scores` on GPT-OSS.** nnterp 1.x reads the softmax input, one key wider than the pattern, tag `sink` (`ru.py:1507-1510`); nnterp reads `attn_weights_1`, the masked scores before the sink column, same width as the pattern (`fam/gpt_oss.py:40-42`; `families.md:136`). `softmax(scores)` reproduces the pattern in neither without handling the sink.

15. **Falcon `mlp_output` is a copy** in nnterp (`fam/falcon.py:136-148`) because the block adds the attention into the MLP tensor in place; nnterp 1.x's `mlps_output[i]` is the live tensor and must be cloned as reached (`test_block_invariants.py:47-50`). Migration: drop the clone.

16. **`skip_layers` signature.** `skip_layers(start, end, skip_with)`; `layer_returns_tuple` gone; negative indices allowed (`S.py:215-229`). Tuple filler is `(hidden, None)` not `(hidden, DummyCache())` (`comp/layer.py:56`). `skip_layer(i)` → `skip_layers(i, i)`.

17. **`steer` signature.** `steer(layers, vector, factor=1.0, token_positions=None, batch_index=None)`; `positions` gone; `steering_vector` positional name is `vector`; no ascending sort (`S.py:231-250`). Migration: rename the kwarg, pass layers ascending.

18. **`get_topk_closest_tokens` / `probs_to_dict`.** Return is always `list[dict]` *(ran)*; tokens are `tokenizer.decode(id)` strings rather than `convert_ids_to_tokens` (`S.py:271-283`); `probs_to_dict(probs, k)` replaces `probs_to_dict(tokens, probs)`.

19. **Tokenizer handling.** `add_prefix_false_tokenizer` and `tokenizer_kwargs` identical. `get_first_tokens` no longer accepts a raw `TransformersModel` and no longer falls back to `model.tokenizer` (`nnterp/prompt_utils.py:39-42`).

20. **`run_prompts` progress bar** off by default (`tqdm=None`, `nnterp/prompt_utils.py:131`).

21. **Errors.** `RenamingError` → `Unavailable` (declared reason), `UnsupportedFamily` (unknown `model_type`), nnsight `SourceNotAvailable` (op moved), `AttributeError` (read-only assign). An out-of-order read of a source-located value in nnterp is an `OutOfOrderError` naming the call's `.fn`, the drill's location, not the value (`docs/developing/gotchas.md`, "An out-of-order read"); nnterp 1.x's `Internals.rank` was the tool to avoid this.

22. **DBRX `self_attn`.** nnterp 1.x aliases `norm_attn_norm` (`ru.py:203`) so `attentions[i]`/`attentions_input[i]` are the container and its (pre-norm) input, with `attentions_output` redirected to `self_attn.attn` (`:1476`); nnterp aliases `norm_attn_norm.attn` (`fam/dbrx.py:24`) so `self_attn.input` is the normed tensor. Migration: `attentions_input[i]` on DBRX → `layers[i].norm_attn_norm.input`.

23. **Import order.** nnterp documents that `import nnterp` (or nnsight) must precede any `transformers.models...modeling_*` import or the process segfaults (`CLAUDE.md:109`, `README.md:384-387`); nnterp 1.x imports transformers classes at import time itself (`utils.py:26-87`), so it never exposed users to that order.

## 4. New in nnterp (no nnterp 1.x counterpart)

- Families registry: one module per `model_type`, lazy import, `nnterp.families.{lookup, register, known, all_families, REGISTRY}`, `UnsupportedFamily` (`fam/__init__.py`).
- `model.support(layer=None)`, `Standard.values()`, `Standard.support()`, per-block dotted keys (`S.py:287-348`, `comp/standard.py`).
- `Unavailable` and the `unavailable=` predicate on every descriptor; `unavailable("reason")` class-body marker; `needs_eager`, `interface_reason`, `needs_torch_kernels`, `needs_recurrent_routing`, `Attention.off_interface()` (`comp/eproperty.py:21-123`, `attention.py:18-89`, `recurrent.py:146-187`).
- Layouts: fourteen `jaxtyping` aliases (`Residual`, `Logits`, `NextTokenProbs`, `Tokens`, `Queries`, `Keys`, `Values`, `Pattern`, `HeadOutputs`, `LinearQK`, `LinearV`, `Gates`, `State`, `States`), `value.layout`, `value.dims`, `isinstance(tensor, Pattern)` (`comp/eproperty.py:127-150`; `api-quick-reference.md:239-260`).
- `StandardizedProperty` sizes a family may define (`S.py:26-50`; `fam/gpt2.py:63`, `falcon.py:157-167`, `deepseek_v2.py:44-51`, `opt.py:49`, `mpt.py:98`, `bloom.py:107`, `gptj.py:90`).
- Descriptors: one `EProperty(key, description, unavailable, select)` whose key is a path (`"output"`, `"../norm.output"`, `"source.<op>.inputs"`, or a function returning one), `DerivedEProperty`, `seq_first`, `first_tensor`, `rewrap`; `Standard.sourced`, the flag a family sets on an envoy whose forward holds a value read after the call starts (`comp/eproperty.py`, `attention.py:48-57`, `standard.py:34-60`).
- Keys for a forward that branches: `RecurrentMixer.KERNEL`, `per_call`, `pinned` (`comp/recurrent.py:190-244`, `:311-329`).
- `attention_values` on every family (nnterp 1.x had queries/keys/scores/head outputs only).
- Attention interior on GPT-J, BLOOM, MPT, Falcon mapped onto their own ops (`fam/gptj.py:47-77`, `bloom.py:46-86`, `mpt.py:49-78`, `falcon.py:75-124`); nnterp 1.x marks those `_no_interface` (`ru.py:1393-1404`, `:1448`, `:1462`, `:1489`, `:1498`).
- Falcon alibi branch (`by_alibi`, `fam/falcon.py:51-58`) covering queries/keys/values/scores/head outputs; nnterp 1.x covered only the pattern on alibi (`ru.py:1491-1494`).
- DeltaNet per-token state: `state`, `states`, `state_after`, `set_state_after`, `route_kernels` / `route_delta_rule` (`comp/recurrent.py:95-143`, `:352-446`); decode-step kernel chosen by the forward's own test (`use_precomputed_states and seq_len == 1`).
- Assignable `input_ids`, `attention_mask`, `logits` (`S.py:170-184`, `:352-370`).
- `envoys=` extension point; a user `EProperty` appears in `support()` (`test_registry.py:106-118`).
- `Layer.skip_with(hidden)` (`comp/layer.py:47-56`); `Layer.returns_tuple` declared per family.
- Falcon `mlp_output` transform-backed copy (`fam/falcon.py:144-148`).
- Per-family test suite `FamilySuite` (`tests/families/suite.py`, ~35 methods × 32 checkpoints) including contribution identity, renamed-vs-raw equality, causal writes, in-place edits, layout/annotation checks, repr checks; `test_registry.py` (lazy import, refusal, register/override, remote key, sizes read-only); `test_base.py`.
- Docs tree with a `CLAUDE.md` router, `docs/reference/families.md` sweep table, `docs/developing/transformers-compat.md` upgrade procedure, pattern recipes (`docs/patterns/*.md`).
- Typing: `layers: Sequence[Layer]`, `self_attn: Attention`, etc. (`S.py:122-126`, `comp/layer.py:36-40`).
- Silent load: no logger.

## 5. Family coverage

**nnterp (31 modules, 32 pinned checkpoints; `fam/`, `families.md:36-69`)**: gpt2, llama, gpt_neox, mistral, mixtral, qwen2, qwen2_moe, qwen3, qwen3_moe, gemma, gemma2, gemma3_text, gpt_oss, deepseek_v2, deepseek_v3, dbrx, phi, phi3, olmo, olmo2, olmo3, smollm3, stablelm, gptj, bloom, mpt, falcon (7B and 40B; alibi tested on a config copy), opt, qwen3_next, qwen3_5_text, qwen3_5_moe_text. Anything else raises `UnsupportedFamily` (`test_registry.py:39-41`: `glm4`).

**nnterp 1.x**: (a) name lists applied to every model (`ru.py:159-236`), so any checkpoint spelling its containers `model/transformer/gpt_neox/decoder/language_model`, its blocks `h/blocks/layers`, its attention `attn/self_attention/attention/norm_attn_norm`, its MLP `mlp/block_sparse_moe/feed_forward/ffn` loads with no code and is validated at load; (b) 12 explicit `FAMILY_ADDRESSES` entries (`:1422-1512`) plus `get_block_structure` naming `gptj, phi, codegen, olmo2` and the `use_parallel_residual/parallel_attn/new_decoder_architecture` flags (`:1515-1532`). Pinned in tests (`tests/test_config.yaml:66-127` + `hybrid_models`): 26 invariant families = nnterp's list minus olmo3, smollm3, plus 3 hybrids; `llama_like_models` adds llama-4 (`llama4` text), phi-3.5-moe (`phimoe`), mistral-nemo, deepseek-llm, gemma-3-34M, qwen1.5/2.5, sbintuitions/tiny-lm-chat; `core_test_models` adds `bigscience-small-testing` (BLOOM `slow_but_exact`); the toy-model collection sweep (~200 repos, `data/toy_models_cache.json`) minus `skip_patterns` (mamba, bamba, lfm2, granite-4.0-h, jamba, falcon-h1, gemma-3n, llama-3.2-vision, bert, whisper).

**nnterp 1.x supports, nnterp lacks** (by `model_type`): `llama4` (text tower via `text_only`), `phimoe`, `codegen`, `glm4`/`glm4v`, `granite` and every other Llama-named family without a module (Qwen 1.5 is `qwen2`, fine), `mllama` (with `allow_multimodal`), VLMs generally (`qwen2_vl`, `llava`, `gemma3` multimodal), Seq2Seq via `detect_automodel`, vLLM. BLOOM `slow_but_exact` loads in both (nnterp's contributions even work there).

**nnterp supports, nnterp 1.x lacks a pinned test for**: `olmo3`, `smollm3` (both load in nnterp 1.x through the lists; untested there). The attention interior on GPT-J/BLOOM/MPT/Falcon and `attention_values` everywhere are nnterp-only (§4).

## 6. Fragile spots in nnterp (constraints, from nnterp 1.x's experience)

1. **Op names are a transformers-version fact.** `INTERFACE = "attention_interface_1"` (`comp/attention.py:28`), `nn_functional_softmax_0`, `nn_functional_dropout_0` (`:118`, `:144`), GPT-OSS `attn_weights_1` (`fam/gpt_oss.py:40`), GPT-J `self__attn_0.source.self_attn_dropout_0` (`fam/gptj.py:72`), BLOOM `self__reshape_0`/`F_softmax_0`/`torch_bmm_0`/`dropout_add_0`/`self_attention_dropout_0` (`fam/bloom.py:46-86`), MPT `query_states_0`/`torch_matmul_1`/`F_dropout_0` (`fam/mpt.py:49-88`), Falcon `apply_rotary_pos_emb_0`/`value_layer_0`/`F_softmax_0|1`/`attn_output_1`/`flatten_0`/`self_attention_dropout_0` (`fam/falcon.py:75-124`), DeltaNet `torch_chunk_gated_delta_rule_0`/`torch_recurrent_gated_delta_rule_0`/`use_precomputed_states_0`/`last_recurrent_state_3` (`comp/linear_attention.py:45-49`). nnterp 1.x was broken twice this way (`_0`→`_1` under nnsight 0.8; `module_attn_dropout_0`→`nn_functional_dropout_0` under transformers 5; CHANGELOG `:203-217`) and DBRX/Qwen2-MoE moved onto the interface in 5.17 (`:182-187`). Constraint: `suite.py:319-334` must stay in CI for every family on every transformers bump; `docs/developing/transformers-compat.md` is the procedure. `last_recurrent_state_3` is an occurrence count inside the kernel body and will move with any edit to that loop.

2. **`.source` snapshots module globals at first drill.** nnsight builds the instrumented forward over a copy of the globals (nnterp 1.x `ru.py:1052-1058`; nnterp `gotchas.md` "`.source` snapshots module globals"). Constraint: `route_delta_rule` before the first trace of a linear block (`comp/recurrent.py:95-129`); any other runtime kernel/monkeypatch switch is invisible after the first trace. nnterp 1.x pinned at load and re-pinned on `dispatch()`; nnterp has no re-pin, which is fine only because `route_kernels` rebinds the module globals themselves rather than swapping and restoring.

3. **`fla` / `causal_conv1d` installed = no DeltaNet values.** nnterp reports them unavailable (`needs_torch_kernels`); nnterp 1.x sourced under the reference kernels and restored the globals. Constraint: a hybrid experiment on a GPU box with `fla` installed gets nothing but `attention_output`; `contributing.md:200-203` lists this as open.

4. **Reads are live objects.** `layer_output.save()` returns the model's tensor. Constraint: on any family whose forward mutates a tensor after the read point, clone in the block. nnterp handles the one known case (Falcon MLP, `fam/falcon.py:136-148`) and DeltaNet `state_input` (`comp/linear_attention.py:85`); `attention_output` on Falcon (`x + attn` added into the MLP tensor, not into `attn`) is safe, but a new family with an in-place `residual.add_()` would be a silent trap.

5. **In-place edits on views.** GPT-2 and MPT q/k/v are split/chunk views; torch refuses in-place (`REFUSES_IN_PLACE_QKV`, `families.md:168`). GPT-2 head outputs are a transposed view that torch accepts today (`comp/attention.py:168-171`). Constraint: assign, do not edit, for q/k/v on those two; any family that returns a non-contiguous view from its interface changes the answer.

6. **`reorder_and_upcast_attn`** is detected in nnterp (`fam/gpt2.py:46-50`) — an improvement over nnterp 1.x, which did not detect it. Constraint: it is the only config flag that switches an attention path that nnterp checks; `scale_attn_by_inverse_layer_idx` and similar do not change the op path but change what the scores mean.

7. **Layer-0-only assumptions.** `support()` builds its key set from every block (`S.py:324-335`), better than nnterp 1.x's probe-layer checks (`ru.py:1682`, `:1931`). But `_hosts` keys on `_standard_children`, which maps by alias/native name; a family whose first block is a different class from the rest (Mllama cross-attention, DeepSeek dense-vs-MoE MLP is handled by keying both classes to `Mlp`, `fam/deepseek_v2.py:39`) would show a `Standard`-less block silently missing values. Constraint: heterogeneous block classes need both classes in `ENVOYS`; nothing refuses a block that is not a `Layer` (`contributing.md:185-189`).

8. **Sizes on non-Llama configs.** `hidden_size`/`num_attention_heads`/`vocab_size` rely on transformers' `attribute_map` aliases (GPT-2 `n_embd`, BLOOM `n_head`, MPT `d_model`, DBRX `d_model`/`n_heads`) *(ran: all resolve)*, but `intermediate_size` does not exist on `DbrxConfig` and raises *(ran)*. Constraint: every family whose config lacks a plain key needs a `def <size>(model)`; the suite's `test_sizes_match_the_model` (`suite.py:557-575`) reads `MLP_WIDTH_KEY` for DBRX/Qwen3-MoE instead of `model.intermediate_size`, so it does not catch this.

9. **Padding-side and last-token assumptions.** `next_token_probs` takes `[:, -1]` (`S.py:204`) and the helpers require left padding for a negative index (`nnterp/nnsight_utils.py:25-30`); same as nnterp 1.x (`st.py:415-419`, `nnsight_utils.py:291-298`). Constraint: `tokenizer_kwargs={"padding_side": "left"}` for batched prompts on both.

10. **Half-precision tolerances.** nnterp 1.x's pattern check widens to `atol=1e-2` under bf16 (`ru.py:960`) and its invariant suite loads in fp32 because the Falcon checkpoints are bf16 and the additive identity holds only to one ulp (`test_block_invariants.py:39-42`). nnterp's suite loads in the checkpoint dtype with `8*eps` tolerances (`suite.py:219-220`, `:284-288`) and forces fp32 only for DBRX (`families.md:164`). Constraint: on a bf16 checkpoint the contribution identity and the row-sum check are one-ulp statements; a real bf16 Falcon/Gemma checkpoint may need `dtype=torch.float32` for the identity to hold to the suite's tolerance. Neither package handles fp16 NaN patterns (softmax over `-inf` rows under fp16 mask) specially; nnterp 1.x has no such handling either.

11. **The kernel choice is cached per call.** `per_call` keeps the choice as `(call, value)` per `(envoy.path, key)` on the worker greenlet (`comp/recurrent.py:190-228`): the call is the step a read is pinned to by `tracer.iter`, and, relaxed, on step 0 or outside `tracer.iter`, the worker's count of passes of the module's `.output`. A second call of the block has another count and a second invoke (or a replayed `model.edit`) another worker, so the record is made again for each. nnterp 1.x's `delta_rule_call` had the mirror-image limit (step 0 = chunked, warm-cache trace unsupported, `ru.py:1023-1029`).

12. **`hasattr` raising `Unavailable`** is a known trap (`comp/eproperty.py:28-32` TODO) that also breaks any third-party code doing `getattr(envoy, name, default)` inside a trace; `contributing.md:204-207` lists a `require/available` helper as open.

13. **Out-of-order reads of source-located values warn instead of raising** (`gotchas.md` "An out-of-order read"), so a saved name silently goes unbound. nnterp 1.x raised on the same situation (or ranked reads via `Internals.rank`). Constraint: check every saved name after a trace that touches the attention interior.

14. **`Attention.SINK`** is documented as an attribute (`api-quick-reference.md:163`) but the base class has no default *(ran: absent)*; only `fam/gpt_oss.py:38` sets it. Constraint: read it as `getattr(type(attn), "SINK", False)`.
