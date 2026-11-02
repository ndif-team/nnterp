# API at a glance

Every public symbol of nnterp, with the page that explains it and the generated page that holds its signature; `model` is a `StandardizedTransformer` unless a row says otherwise.

## The model classes

The constructors of [`StandardizedTransformer`](../documentation/standardized.md#nnterp.standardized.StandardizedTransformer) and [`StandardizedVLLM`](../documentation/standardized_vllm.md#nnterp.standardized_vllm.StandardizedVLLM), which share the [`Standardized`](../documentation/standardized.md#nnterp.standardized.Standardized) mixin, take nnsight's keywords plus these.

| keyword | `StandardizedTransformer` | `StandardizedVLLM` | page |
| --- | --- | --- | --- |
| `repo_id` | a Hub repo id, or an already-loaded `torch.nn.Module` | a Hub repo id | [Loading](../guide/loading.md#an-already-loaded-module) |
| `family=` | the family for this load, a module or any object with `RENAME` and `ENVOYS`; the registry is not consulted | the same, written against vLLM's module classes | [Registering](../developing/registering.md#one-load) |
| `rename=` | extra aliases, merged over the family's `RENAME`; a key given here wins | the same | [Loading](../guide/loading.md#rename-and-envoys) |
| `envoys=` | extra entries over nnsight's tensor-parallel envoys and the family's `ENVOYS`; a key given here wins | the same, over nnsight's envoys for vLLM's parallel layers | [Custom values](../developing/custom-values.md#key-on-the-module-type) |
| `tokenizer_kwargs=` | attributes set on the loaded tokenizer, such as `padding_side` and `pad_token` | not taken | [Loading](../guide/loading.md#tokenizer_kwargs) |
| `**kwargs` | passed to `TransformersModel`: `dispatch`, `device`, `dtype`, `attn_implementation`, `revision`, `trust_remote_code`, `task` (default `"text-generation"`) | passed to nnsight's `VLLM`: `dispatch` and the engine's arguments | [Loading](../guide/loading.md#arguments), [vLLM](../vllm/index.md) |

## Methods and attributes

| member | what | where | page |
| --- | --- | --- | --- |
| [`support(layer=None)`](../documentation/standardized.md#nnterp.standardized.Standardized.support) | which values this checkpoint has: `None`, or the reason, per block without `layer` | outside a trace; nothing runs | [Availability](../guide/values/availability.md) |
| [`order(layer=None)`](../documentation/standardized.md#nnterp.standardized.StandardizedTransformer.order), [`rank(name, layer=None)`](../documentation/standardized.md#nnterp.standardized.StandardizedTransformer.rank) | the forward order of the available values, name to rank; one value's `(side, rank)` | outside; one probe run on first use; `StandardizedTransformer` only | [Attention: read order](../guide/values/attention.md#read-order) |
| [`skip_layers(start, end, skip_with=None)`](../documentation/standardized.md#nnterp.standardized.Standardized.skip_layers) | blocks `start` to `end` inclusive do not run | inside a trace, before block `start` | [Methods](../guide/methods.md#skip_layers) |
| [`steer(layers, vector, factor=1.0, token_positions=None, batch_index=None)`](../documentation/standardized.md#nnterp.standardized.Standardized.steer) | adds `factor * vector` to `layer_output` in place | inside; `layers` ascending | [Methods](../guide/methods.md#steer) |
| [`project_on_vocab(hidden)`](../documentation/standardized.md#nnterp.standardized.StandardizedTransformer.project_on_vocab) | the logit lens: the final norm, `lm_head`, and what the model does past the head | inside on a live value, outside on a saved one; inside only on vLLM | [Methods](../guide/methods.md#project_on_vocab) |
| [`get_topk_closest_tokens(hidden, k=5)`](../documentation/standardized.md#nnterp.standardized.Standardized.get_topk_closest_tokens) | `project_on_vocab`, softmax, the `k` most likely tokens per position | either | [Methods](../guide/methods.md#get_topk_closest_tokens-and-probs_to_dict) |
| [`probs_to_dict(probs, k=5)`](../documentation/standardized.md#nnterp.standardized.Standardized.probs_to_dict) | the `k` most likely tokens of one `[vocab_size]` distribution | either | [Methods](../guide/methods.md#get_topk_closest_tokens-and-probs_to_dict) |
| `to_model_key()` | nnsight's remote key, naming `TransformersModel` (or `VLLM`) | outside | [Remote](../guide/remote.md) |
| `num_layers`, `hidden_size`, `vocab_size`, `num_heads`, `num_kv_heads`, `head_dim`, `qk_head_dim`, [`intermediate_size`](../documentation/standardized.md#nnterp.standardized.Standardized.intermediate_size) | the sizes, read off the config and read-only; a family defines `def <size>(model)` to spell one otherwise | outside; no `dispatch` needed | [The whole model](../guide/values/whole-model.md#sizes) |
| `layers[i].self_attn.num_heads`, `.num_kv_heads`, `.head_dim`, `.qk_head_dim`; `layers[i].mlp.intermediate_size`; `layers[i].mlp.num_experts`, `.top_k` | a block's own sizes, read off the module | either | [The whole model](../guide/values/whole-model.md#sizes) |
| [`family`](../documentation/standardized.md#nnterp.standardized.StandardizedTransformer.family) | the family module the load resolved to, or the `family=` passed | | [Loading](../guide/loading.md) |
| `layers`, `embed_tokens`, `norm`, `lm_head`; `layers[i].self_attn`, `.mlp`, `.linear_attn` | the standard names: the root's modules, and a block's `Attention`, `Mlp` and recurrent mixer | | [Names](../guide/names.md) |
| [`vision`](../documentation/standardized.md#nnterp.standardized.StandardizedTransformer.vision), [`projector`](../documentation/standardized.md#nnterp.standardized.StandardizedTransformer.projector) | the vision tower and the last module before the scatter, on a wrapper whose family names them | | [Vision-language](../guide/architectures/vision-language.md) |
| [`add_prefix_false_tokenizer`](../documentation/standardized.md#nnterp.standardized.StandardizedTransformer.add_prefix_false_tokenizer) | the tokenizer loaded with `add_prefix_space=False`, on first use | outside | [Helpers](../guide/helpers.md#target-token-mass-over-many-prompts) |

## Root values

The whole-model values, in forward order; the layouts are `Logits`, `NextTokenProbs` and `Tokens` from `nnterp.standardized` and `Residual` from `nnterp.components`.

| value | layout | writes | on `StandardizedVLLM` | page |
| --- | --- | --- | --- | --- |
| [`input_ids`](../documentation/standardized.md#nnterp.standardized.StandardizedTransformer.input_ids) | `Tokens`, `[batch, seq]` | assign | read-only, `[1, tokens]` | [The whole model](../guide/values/whole-model.md#the-input) |
| [`attention_mask`](../documentation/standardized.md#nnterp.standardized.StandardizedTransformer.attention_mask) | `Tokens` | assign | unavailable | [The whole model](../guide/values/whole-model.md#the-input) |
| [`input_size`](../documentation/standardized.md#nnterp.standardized.StandardizedTransformer.input_size) | a `torch.Size` | read-only | read-only | [The whole model](../guide/values/whole-model.md#the-input) |
| [`token_embeddings`](../documentation/standardized.md#nnterp.standardized.StandardizedTransformer.token_embeddings) | `Residual` | assign | `[1, tokens, hidden_size]` | [The whole model](../guide/values/whole-model.md#token_embeddings-and-block-0) |
| [`final_norm_output`](../documentation/standardized.md#nnterp.standardized.StandardizedTransformer.final_norm_output) | `Residual` | assign, in place | `[1, tokens, hidden_size]` | [The whole model](../guide/values/whole-model.md) |
| [`lm_head_output`](../documentation/standardized.md#nnterp.standardized.StandardizedTransformer.lm_head_output) | `Logits` | assign, in place | unavailable | [The whole model](../guide/values/whole-model.md#logits-and-lm_head_output) |
| [`logits`](../documentation/standardized.md#nnterp.standardized.StandardizedTransformer.logits) | `Logits` | assign | `[1, 1, vocab_size]`, this step's last position | [The whole model](../guide/values/whole-model.md#logits-and-lm_head_output) |
| [`next_token_probs`](../documentation/standardized.md#nnterp.standardized.StandardizedTransformer.next_token_probs) | `NextTokenProbs`, `[batch, vocab_size]` | read-only | `[1, vocab_size]` | [The whole model](../guide/values/whole-model.md#next_token_probs) |

## Block values

| value | host | layout | available when | page |
| --- | --- | --- | --- | --- |
| [`layer_input`](../documentation/components/layer.md#nnterp.components.layer.Layer.layer_input) | `layers[i]` | `Residual` | always | [Residual stream](../guide/values/residual-stream.md) |
| [`attention_norm_output`](../documentation/components/layer.md#nnterp.components.layer.Layer.attention_norm_output) | `layers[i]` | `Residual` | the block has an `input_layernorm` | [Residual stream](../guide/values/residual-stream.md#between-the-sublayers) |
| [`layer_mid`](../documentation/components/layer.md#nnterp.components.layer.Layer.layer_mid) | `layers[i]` | `Residual` (`FlatResidual` on OPT) | a stream between the sublayers: not on a parallel block | [Residual stream](../guide/values/residual-stream.md#between-the-sublayers) |
| [`mlp_norm_output`](../documentation/components/layer.md#nnterp.components.layer.Layer.mlp_norm_output) | `layers[i]` | `Residual` (`FlatResidual` on OPT) | the block norms the MLP's input | [Residual stream](../guide/values/residual-stream.md#between-the-sublayers) |
| [`mlp_input`](../documentation/components/layer.md#nnterp.components.layer.Layer.mlp_input) | `layers[i]` | `Residual` | the block has an `mlp` | [Residual stream](../guide/values/residual-stream.md#between-the-sublayers) |
| [`layer_output`](../documentation/components/layer.md#nnterp.components.layer.Layer.layer_output) | `layers[i]` | `Residual` (`Streams` on DeepSeek-V4) | always | [Residual stream](../guide/values/residual-stream.md) |
| [`attention_output`](../documentation/components/attention.md#nnterp.components.attention.Attention.attention_output) | `self_attn` | `Residual` | always | [Residual stream](../guide/values/residual-stream.md) |
| [`attention_queries`](../documentation/components/attention.md#nnterp.components.attention.Attention.attention_queries) | `self_attn` | `Queries` | eager attention | [Attention](../guide/values/attention.md) |
| [`attention_keys`](../documentation/components/attention.md#nnterp.components.attention.Attention.attention_keys) | `self_attn` | `Keys` | eager attention | [Attention](../guide/values/attention.md) |
| [`attention_values`](../documentation/components/attention.md#nnterp.components.attention.Attention.attention_values) | `self_attn` | `Values` | eager attention | [Attention](../guide/values/attention.md) |
| [`attention_scores`](../documentation/components/attention.md#nnterp.components.attention.Attention.attention_scores) | `self_attn` | `Pattern` | eager attention | [Attention](../guide/values/attention.md) |
| [`attention_probabilities`](../documentation/components/attention.md#nnterp.components.attention.Attention.attention_probabilities) | `self_attn` | `Pattern` | eager attention | [Attention](../guide/values/attention.md) |
| [`attention_head_outputs`](../documentation/components/attention.md#nnterp.components.attention.Attention.attention_head_outputs) | `self_attn` | `HeadOutputs` | eager attention | [Attention](../guide/values/attention.md) |
| [`attention_premix`](../documentation/components/attention.md#nnterp.components.attention.Attention.attention_premix) | `self_attn` | `HeadOutputs` | the attention has an `o_proj`; any implementation | [Residual stream](../guide/values/residual-stream.md#between-the-sublayers) |
| [`mlp_output`](../documentation/components/mlp.md#nnterp.components.mlp.Mlp.mlp_output) | `mlp` | `Residual` | the block has an `mlp` | [Residual stream](../guide/values/residual-stream.md) |
| [`mlp_activation`](../documentation/components/mlp.md#nnterp.components.mlp.Mlp.mlp_activation) | `mlp` | `Neurons` | the MLP has an `act_fn`; not on a mixture | [Residual stream](../guide/values/residual-stream.md#between-the-sublayers) |
| [`mlp_neurons`](../documentation/components/mlp.md#nnterp.components.mlp.Mlp.mlp_neurons) | `mlp` | `Neurons` | the MLP has a `down_proj`; not on a mixture | [Residual stream](../guide/values/residual-stream.md#between-the-sublayers) |
| [`router_logits`](../documentation/components/moe.md#nnterp.components.moe.Moe.router_logits) | `mlp`, a `Moe` | `RouterLogits` | the module runs a mixture | [Mixture of experts](../guide/architectures/mixture-of-experts.md) |
| [`expert_weights`](../documentation/components/moe.md#nnterp.components.moe.Moe.expert_weights), [`expert_indices`](../documentation/components/moe.md#nnterp.components.moe.Moe.expert_indices) | `mlp`, a `Moe` | `ExpertWeights`, `ExpertIndices` | the module runs a mixture | [Mixture of experts](../guide/architectures/mixture-of-experts.md) |
| [`expert_outputs`](../documentation/components/moe.md#nnterp.components.moe.Moe.expert_outputs) | `mlp`, a `Moe` | `ExpertOutputs` | `experts_implementation` is `grouped_mm` or `batched_mm` | [Mixture of experts](../guide/architectures/mixture-of-experts.md#expert_outputs-needs-the-grouped-experts) |
| [`routed_output`](../documentation/components/moe.md#nnterp.components.moe.Moe.routed_output) | `mlp`, a `Moe` | `Residual` | the module runs a mixture | [Mixture of experts](../guide/architectures/mixture-of-experts.md) |
| [`shared_expert_output`](../documentation/components/moe.md#nnterp.components.moe.Moe.shared_expert_output) | `mlp`, a `Moe` | `Residual` | the mixture has a shared expert | [Mixture of experts](../guide/architectures/mixture-of-experts.md#read-order) |

## Recurrent mixer values

The values of `layers[i].linear_attn` on a [gated DeltaNet](../guide/architectures/delta-net.md) (`LinearAttention`), a [Mamba-1](../guide/architectures/mamba.md) (`SelectiveScan`) and a [Mamba-2](../guide/architectures/mamba-2.md) (`StateSpace`) mixer, with the kernel's own name for the tensor where it has one.

| value | [`LinearAttention`](../documentation/components/linear_attention.md#nnterp.components.linear_attention.LinearAttention) | [`SelectiveScan`](../documentation/components/selective_scan.md#nnterp.components.selective_scan.SelectiveScan) | [`StateSpace`](../documentation/components/state_space.md#nnterp.components.state_space.StateSpace) | available when |
| --- | --- | --- | --- | --- |
| `attention_output` | `Residual` | `Residual` | `Residual` | always |
| `attention_queries` | `LinearQK` | `ScanQK` (`C`) | `SSDQueries` (`C`) | the kernels are transformers' pure-torch ones |
| `attention_keys` | `LinearQK` | `ScanQK` (`B`) | `SSDKeys` (`B`) | pure-torch kernels |
| `attention_values` | `LinearV` | `ScanValues` (`x`) | `SSDValues` (`x`) | pure-torch kernels |
| `betas` | `Gates` | `ScanSteps` (`dt`), read-only | `Gates` (`dt`) | pure-torch kernels |
| `decays` | `Gates` (`ChannelGates` on Kimi-Linear) | `ScanDecays` (`dt * A`), read-only | `Gates` (`A * dt`) | pure-torch kernels |
| `state_input` | `State`, or `None` on a prompt | `ScanState`, or `None` on a prompt | `State`, or `None` on a prompt | pure-torch kernels |
| `attention_head_outputs` | `LinearV` | `ScanValues` (`y`) | `SSDHeadOutputs` (`y`) | pure-torch kernels, read inside them on Mamba-1 |
| `state_output` | `State` | `ScanState` | `State` | pure-torch kernels, read inside them on Mamba-1 |
| [`state`](../documentation/components/recurrent.md#nnterp.components.recurrent.RecurrentMixer.state) | `State`, one occurrence per token | `ScanState`, one per token | unavailable | `route_kernels(family, "torch")` |
| [`states`](../documentation/components/recurrent.md#nnterp.components.recurrent.RecurrentMixer.states) | `States`, read-only | `ScanStates`, read-only | `States`, read-only | `route_kernels`; on Mamba-2 `chunk_per_token(model)` |
| [`state_after(t)`](../documentation/components/recurrent.md#nnterp.components.recurrent.RecurrentMixer.state_after), [`set_state_after(t, value)`](../documentation/components/recurrent.md#nnterp.components.recurrent.RecurrentMixer.set_state_after) | both | both | `state_after` only | as `states` |

## Vision values

The values of `model.vision` on a wrapper loaded with `task="image-text-to-text"`, on the [Vision-language](../guide/architectures/vision-language.md) page; `vision.support()` and `vision.no_images()` answer for them.

| value | layout | read at | available when |
| --- | --- | --- | --- |
| [`image_token_mask`](../documentation/components/vision.md#nnterp.components.vision.Vision.image_token_mask) | `ImageTokenMask`, `[batch, seq]` bool | the model's inputs, so first in a trace; read-only | an image reaches the model and the config names an `image_token_id` |
| [`patch_embeddings`](../documentation/components/vision.md#nnterp.components.vision.Vision.patch_embeddings) | `Patches` | `patch_embed.output`, one row per patch | an image reaches the model |
| [`tower_output`](../documentation/components/vision.md#nnterp.components.vision.Vision.tower_output) | `Patches` | the last block's stream after `vision.norm` | an image reaches the model |
| [`image_features`](../documentation/components/vision.md#nnterp.components.vision.Vision.image_features) | `ImageFeatures`, `[image_tokens, hidden_size]` | the scatter into the token embeddings | an image reaches the model and the family keys an `ImageScatter` |
| `vision.layers[i].layer_output` and the other block values | `Patches` for the stream values; the attention interior as on a text block | the tower's blocks ([`VisionLayer`](../documentation/components/vision.md#nnterp.components.vision.VisionLayer), [`VisionAttention`](../documentation/components/vision.md#nnterp.components.vision.VisionAttention), [`VisionMlp`](../documentation/components/vision.md#nnterp.components.vision.VisionMlp)) | an image reaches the model; eager attention for the interior |
| `vision.num_layers`, `vision_hidden_size`, `num_heads`, `head_dim`, `intermediate_size`, `patch_size`, `image_size` | sizes, off the tower's config | outside a trace | `image_size` raises `Unavailable` on a tower with no fixed resolution |

## Helpers

| function | what | page |
| --- | --- | --- |
| [`nnsight_utils.get_token_activations(model, prompts, layers=None, get_activations=None, remote=False, idx=None, tracer=None)`](../documentation/nnsight_utils.md#nnterp.nnsight_utils.get_token_activations) | `[len(layers), num_prompts, hidden_size]` at token `idx` (default `-1`) of every prompt | [Helpers](../guide/helpers.md) |
| [`nnsight_utils.collect_token_activations_batched(model, prompts, batch_size, ..., tqdm=None, use_session=True)`](../documentation/nnsight_utils.md#nnterp.nnsight_utils.collect_token_activations_batched) | the same over batches; a remote run goes through one session | [Helpers](../guide/helpers.md#another-value-many-prompts) |
| [`nnsight_utils.collect_last_token_activations_session(model, prompts, batch_size, ...)`](../documentation/nnsight_utils.md#nnterp.nnsight_utils.collect_last_token_activations_session) | the same inside one `model.session` | [Remote](../guide/remote.md#decided-by-the-deployment) |
| [`nnsight_utils.compute_next_token_probs(model, prompt, remote=False)`](../documentation/nnsight_utils.md#nnterp.nnsight_utils.compute_next_token_probs) | `next_token_probs` per prompt, `[num_prompts, vocab_size]` on the CPU | [Helpers](../guide/helpers.md#another-value-many-prompts) |
| [`nnsight_utils.layer_output(model, layer)`](../documentation/nnsight_utils.md#nnterp.nnsight_utils.layer_output) | the default `get_activations`: `model.layers[layer].layer_output` | [Helpers](../guide/helpers.md) |
| [`prompt_utils.Prompt`](../documentation/prompt_utils.md#nnterp.prompt_utils.Prompt): `from_strings(prompt, target_strings, model_or_tokenizer)`, `has_no_collisions()`, `get_target_probs(probs)`, `run(model, get_probs)` | a prompt with named sets of target token ids | [Helpers](../guide/helpers.md#target-token-mass-over-many-prompts) |
| [`prompt_utils.get_first_tokens(words, model_or_tokenizer)`](../documentation/prompt_utils.md#nnterp.prompt_utils.get_first_tokens) | the first token of `word` and of `" word"`, through `add_prefix_false_tokenizer` | [Helpers](../guide/helpers.md#target-token-mass-over-many-prompts) |
| [`prompt_utils.run_prompts(model, prompts, batch_size=32, get_probs_func=None, func_kwargs=None, remote=False, tqdm=None)`](../documentation/prompt_utils.md#nnterp.prompt_utils.run_prompts) | each target's mass per prompt, `[num_prompts, layers]` | [Helpers](../guide/helpers.md#target-token-mass-over-many-prompts) |
| [`prompt_utils.next_token_probs_unsqueeze(model, prompt, remote=False)`](../documentation/prompt_utils.md#nnterp.prompt_utils.next_token_probs_unsqueeze) | the default `get_probs_func`, with a layer axis of one | [Helpers](../guide/helpers.md#target-token-mass-over-many-prompts) |
| [`prompt_utils.TokenizationError`](../documentation/prompt_utils.md#nnterp.prompt_utils.TokenizationError) | a word has no standalone first token under the tokenizer | [Helpers](../guide/helpers.md#target-token-mass-over-many-prompts) |

## Families, kernels and exceptions

| name | what | page |
| --- | --- | --- |
| [`families.lookup(model_type, engine=None)`](../documentation/families/index.md#nnterp.families.lookup) | the family: a registered one, else `nnterp.families.<model_type>` imported on first use, else `default` with a warning; on another engine `UnsupportedFamily` | [Registering](../developing/registering.md#the-lookup-order) |
| [`families.register(family, *model_types, engine=None)`](../documentation/families/index.md#nnterp.families.register) | a family for every later load of those types, or of the type its `__name__` ends in; also overrides a shipped one | [Registering](../developing/registering.md) |
| [`families.known(engine=None)`](../documentation/families/index.md#nnterp.families.known), [`families.all_families(engine=None)`](../documentation/families/index.md#nnterp.families.all_families) | the shipped model types (98; 60 on vLLM); every shipped family, imported | [Loading](../guide/loading.md) |
| [`families.REGISTRY`](../documentation/families/index.md#nnterp.families.REGISTRY), `families.default`, `families.<model_type>` | what `register` added; the best-effort family; a family module, imported on first access | [Architecture](../developing/internals/architecture.md#from-model_type-to-a-family) |
| [`route_kernels(family, kernel="torch")`](../documentation/components/recurrent.md#nnterp.components.recurrent.route_kernels) | rebind a family's recurrent kernels, process-wide, to transformers' pure-torch functions, or `"default"` back | [Gated DeltaNet](../guide/architectures/delta-net.md#kernels-and-route_kernels) |
| [`route_delta_rule(family, kernel="recurrent")`](../documentation/components/recurrent.md#nnterp.components.recurrent.route_delta_rule) | `route_kernels` in the delta rule's words: `"recurrent"` or `"chunked"` | [Recurrent mixers](../developing/internals/recurrent-mixers.md#pure-torch-kernels-and-routing) |
| [`chunk_per_token(model, enabled=True)`](../documentation/components/state_space.md#nnterp.components.state_space.chunk_per_token) | set every Mamba-2 mixer's `chunk_size` to 1 on this model, so `states` reads every token | [Mamba-2](../guide/architectures/mamba-2.md#the-state-after-every-token) |
| [`axis_size(envoy, axis)`](../documentation/components/standard.md#nnterp.components.standard.axis_size) | the size of a layout axis for a value on `envoy`; `ValueError` for an axis that is not a fact about the model | [Layouts](../guide/values/layouts.md#axes-named-as-sizes) |
| [`Unavailable`](../documentation/components/eproperty.md#nnterp.components.eproperty.Unavailable) | a `RuntimeError` at the read or write of a value the checkpoint lacks, before the model runs | [Availability](../guide/values/availability.md#nnterpunavailable-at-the-read) |
| [`UnsupportedFamily`](../documentation/families/index.md#nnterp.families.UnsupportedFamily) | a `ValueError` at load: no family covers the `model_type` and the default cannot standardize it | [Loading](../guide/loading.md#a-model_type-with-no-family) |
| nnsight's `OutOfOrderError`, `SourceNotAvailable` | a read after the model ran past its location; a `source` path this run does not have | [Attention: read order](../guide/values/attention.md#read-order), [Finding source ops](../developing/finding-source-ops.md#errors-from-a-wrong-name) |

## For family authors

What `nnterp.components` exports beyond the layouts, grouped by role.

| names | what | page |
| --- | --- | --- |
| [`Standard`](../documentation/components/standard.md#nnterp.components.standard.Standard) (`values()`, `support()`, `sourced`), [`Layer`](../documentation/components/layer.md#nnterp.components.layer.Layer) (`returns_tuple`, `skip_with`), [`Attention`](../documentation/components/attention.md#nnterp.components.attention.Attention) (`off_interface()`, `SINK`), [`Mlp`](../documentation/components/mlp.md#nnterp.components.mlp.Mlp), [`Moe`](../documentation/components/moe.md#nnterp.components.moe.Moe) (`SCORING`, `no_mixture()`) | the envoy bases a family subclasses and keys on its module types in `ENVOYS` | [Adding a family](../developing/adding-a-family.md) |
| [`RecurrentMixer`](../documentation/components/recurrent.md#nnterp.components.recurrent.RecurrentMixer) (`CHUNK_KERNEL`, `RECURRENT_KERNEL`, `STATE_OP`, `STEP_STATE_OP`, `BRANCH`, `SEQ_OP`, `KERNEL`), `LinearAttention`, `SelectiveScan`, `StateSpace` | the recurrent mixer bases | [Recurrent mixers](../developing/internals/recurrent-mixers.md) |
| [`Vision`](../documentation/components/vision.md#nnterp.components.vision.Vision), `VisionLayer`, `VisionAttention`, `VisionMlp`, `QwenVision`, `QwenVisionAttention`, `PixtralVision`, [`ImageScatter`](../documentation/components/vision.md#nnterp.components.vision.ImageScatter) | the vision tower's envoys, and the marker class for the module that scatters the image features | [Vision](../developing/internals/vision.md) |
| [`EProperty(key=None, description=None, unavailable=None, select=None)`](../documentation/components/eproperty.md#nnterp.components.eproperty.EProperty), [`DerivedEProperty(compute, description=None, unavailable=None)`](../documentation/components/eproperty.md#nnterp.components.eproperty.DerivedEProperty), [`TokenEProperty`](../documentation/components/tokens.md#nnterp.components.tokens.TokenEProperty) | the descriptors a value is made of; the third serves a tensor flat over tokens as this invoke's rows | [Descriptors](../developing/internals/descriptors.md) |
| [`unavailable(reason)`](../documentation/components/eproperty.md#nnterp.components.eproperty.unavailable) | the marker a family assigns in place of a value it lacks | [Overriding a value](../developing/overriding-values.md#a-value-the-family-does-not-have) |
| `needs_eager`, `interface_reason`, `mixture_reason`, `needs_grouped_experts`, `no_shared_expert`, `needs_torch_kernels`, `needs_recurrent_routing`, `needs_kernel_source`, `needs_token_loop`, `needs_per_token_chunks`, `image_token_id` | `unavailable=` predicates, a function of the envoy returning the reason or `None` | [Overriding a value](../developing/overriding-values.md#a-value-the-family-does-not-have) |
| `PARALLEL`, `POST_NORM_ATTENTION`, `POST_NORM_MLP`, `NOT_ON_INTERFACE`, `PER_IMAGE` | reason strings for the blocks a family marks | [Overriding a value](../developing/overriding-values.md#a-value-the-family-does-not-have) |
| `INTERFACE`, `LOGITS`, `DISPATCH`, `PER_SLOT` | operation names: the shared attention call, the router's projection, the experts call and the per-slot view | [transformers compatibility](../developing/transformers-compatibility.md#the-shared-interface) |
| `first_tensor`, `rewrap`, `seq_first`, `per_call`, `pinned`, `blocks_support` | a tuple output's tensor and its inverse, a heads-first transpose, the once-per-call record, the occurrence pin, what `support()` builds the block rows with | [Descriptors](../developing/internals/descriptors.md#select-and-writes) |
| `BATCH`, `SEQ`, `TOKENS`, `HIDDEN`, `VOCAB`, `HEADS`, `KV_HEADS`, `HEAD_DIM`, `QK_HEAD_DIM`, `INTERMEDIATE`, `QUERY`, `KEY`, `EXPERTS`, `TOP_K`, `STREAMS`, `MIXER_HEADS`, `KEY_DIM`, `VALUE_DIM`, `CHANNELS`, `STATE_DIM`, `GROUPS`, `IMAGES`, `PATCHES`, `IMAGE_TOKENS`, `VISION_HIDDEN` | the axis names every layout is composed from | [Layouts](../guide/values/layouts.md#axes-named-as-sizes) |

## Layouts

The 34 layout names `nnterp.components` exports and the root's three in `nnterp.standardized`, each defined in the generated page of the envoy that serves it.

| layout | axes | carried by | defined in |
| --- | --- | --- | --- |
| `Residual` | `batch seq hidden_size` | the stream values, the contributions, the module boundaries, `token_embeddings`, `final_norm_output`, `routed_output`, `shared_expert_output` | [layer](../documentation/components/layer.md) |
| `FlatResidual` | `tokens hidden_size` | OPT's `layer_mid`, `mlp_norm_output` | [layer](../documentation/components/layer.md) |
| `Streams`, `StreamWeights`, `StreamMixing` | `batch seq streams hidden_size`; `batch seq streams`; `batch seq streams streams` | DeepSeek-V4's `layer_output`, `layer_input`; its `attention_post`, `mlp_post`; its `attention_comb`, `mlp_comb` | [layer](../documentation/components/layer.md) |
| `Logits`, `NextTokenProbs`, `Tokens` | `batch seq vocab_size`; `batch vocab_size`; `batch seq` (integer) | `logits`, `lm_head_output`; `next_token_probs`; `input_ids`, `attention_mask` | [standardized](../documentation/standardized.md) |
| `Queries` | `batch num_heads seq qk_head_dim` | `attention_queries` | [attention](../documentation/components/attention.md) |
| `Keys`, `Values` | `batch num_kv_heads seq qk_head_dim`; `batch num_kv_heads seq head_dim` | `attention_keys`; `attention_values` | [attention](../documentation/components/attention.md) |
| `Pattern` | `batch num_heads query key` | `attention_scores`, `attention_probabilities` | [attention](../documentation/components/attention.md) |
| `HeadOutputs` | `batch seq num_heads head_dim` | `attention_head_outputs`, `attention_premix` | [attention](../documentation/components/attention.md) |
| `Neurons` | `batch seq intermediate_size` | `mlp_activation`, `mlp_neurons` | [mlp](../documentation/components/mlp.md) |
| `RouterLogits` | `batch seq num_experts` | `router_logits` | [moe](../documentation/components/moe.md) |
| `ExpertWeights`, `ExpertIndices` | `batch seq top_k` (`ExpertIndices` integer) | `expert_weights`, `expert_indices` | [moe](../documentation/components/moe.md) |
| `ExpertOutputs` | `batch seq top_k hidden_size` | `expert_outputs` | [moe](../documentation/components/moe.md) |
| `State`, `States` | `batch heads key_dim value_dim`; `batch seq heads key_dim value_dim` | a DeltaNet's or Mamba-2's `state_input`, `state_output`, `state`; `states` | [recurrent](../documentation/components/recurrent.md) |
| `LinearQK`, `LinearV` | `batch seq heads key_dim`; `batch seq heads value_dim` | a DeltaNet's queries, keys; its values, head outputs | [linear_attention](../documentation/components/linear_attention.md) |
| `Gates`, `ChannelGates` | `batch seq heads`; `batch seq heads key_dim` | `decays`, `betas` on a DeltaNet or Mamba-2; Kimi-Linear's `decays` | [linear_attention](../documentation/components/linear_attention.md) |
| `ScanQK` | `batch seq groups state_dim` | a Mamba-1's queries (`C`), keys (`B`) | [selective_scan](../documentation/components/selective_scan.md) |
| `ScanValues`, `ScanSteps` | `batch seq channels` | a Mamba-1's values (`x`), head outputs (`y`); its `betas` | [selective_scan](../documentation/components/selective_scan.md) |
| `ScanDecays` | `batch seq channels state_dim` | a Mamba-1's `decays` | [selective_scan](../documentation/components/selective_scan.md) |
| `ScanState`, `ScanStates` | `batch channels state_dim`; `batch seq channels state_dim` | a Mamba-1's `state_input`, `state_output`, `state`; `states` | [selective_scan](../documentation/components/selective_scan.md) |
| `SSDQueries`, `SSDKeys` | `batch seq groups state_dim` | a Mamba-2's queries (`C`), keys (`B`) | [state_space](../documentation/components/state_space.md) |
| `SSDValues`, `SSDHeadOutputs` | `batch seq heads head_dim` | a Mamba-2's values (`x`), head outputs (`y`) | [state_space](../documentation/components/state_space.md) |
| `Patches` | `images patches vision_hidden_size` | a tower block's stream values, `patch_embeddings`, `tower_output` | [vision](../documentation/components/vision.md) |
| `ImageTokenMask`, `ImageFeatures` | `batch seq` (bool); `image_tokens hidden_size` | `image_token_mask`; `image_features` | [vision](../documentation/components/vision.md) |

The terms these tables use are defined in the [Glossary](glossary.md).
