# Glossary

The terms this site uses, each in the one sense the pages give it, with the page that explains it. nnsight's own terms (envoy, invoke, occurrence, source) are defined only as far as nnterp leans on them, and [nnsight.net](https://nnsight.net) has the rest.

**Alias**: a second attribute that nnsight's `rename=` installs on an envoy, so `model.layers[5]` and GPT-2's `model.transformer.h[5]` are one object and the native name keeps working. Every standard name is one ([Names](../guide/names.md)).

**Attention sink**: a per-head column that joins the softmax and is dropped afterwards (GPT-OSS, MiMo-V2-Flash, DeepSeek-V4), so the pattern's rows sum to less than one ([Attention](../guide/values/attention.md)).

**Availability**: whether this checkpoint has a value. `support()` reports it before any trace, and a read or write of a value the checkpoint lacks raises `Unavailable` with the same reason. The descriptor's `unavailable=`, a reason string or a predicate of the envoy, decides it per envoy ([Availability](../guide/values/availability.md)).

**Axis**: one named dimension of a layout, such as `batch`, `seq` or `num_heads`. An axis that is a fact about the model is named as the size that holds it, and `nnterp.axis_size(envoy, axis)` resolves it ([Layouts](../guide/values/layouts.md#axes-named-as-sizes)).

**Block**: one decoder layer, `model.layers[i]`, a `Layer` envoy. The residual stream enters it as `layer_input`, each sublayer adds its contribution, and it leaves as `layer_output` ([Residual stream](../guide/values/residual-stream.md)).

**Checkpoint**: one Hub repository, such as `openai-community/gpt2`. Each family's test pins a tiny random checkpoint ([Testing](../developing/testing.md)).

**Contribution**: what a sublayer adds to the residual stream, `attention_output` and `mlp_output`, defined by the contribution identity `layer_input + attention_output + mlp_output == layer_output`. On a sandwich or post-norm block it is the post-norm's output, and on BLOOM the tensor entering the residual add ([Residual stream](../guide/values/residual-stream.md)).

**Decode step**: one forward call of `model.generate` after the prompt call, producing one token. Under `tracer.iter`, step 0 is the prompt call and every later step a decode step ([Generation](../guide/generation.md)).

**Default family**: `nnterp.families.default`, the best-effort family that a `model_type` with no family loads with, and a warning. It serves `layer_output`, the root values, the sizes and the attention interior, and refuses the contributions and `project_on_vocab` ([Loading](../guide/loading.md#a-model_type-with-no-family)).

**Derived value**: a `DerivedEProperty`, computed at read time from other served values and read-only: `next_token_probs`, `states`, a Mamba-1 mixer's `betas` and `decays`. Assigning one raises `AttributeError` ([Custom values](../developing/custom-values.md)).

**Dispatch**: `dispatch=True` loads the weights at construction. Without it the model is built on the meta device, with no weights, and loads at the first trace, while `support()` and the sizes read the config either way ([Loading](../guide/loading.md#arguments)).

**Drill**: instrumenting a function that the forward calls, through `<call>.source`, so the operations inside it can be read. It works inside a trace only, and an `EProperty` whose path has two `source` segments drills before every read ([Finding source ops](../developing/finding-source-ops.md#inside-a-call)).

**Eager attention**: transformers' `attn_implementation="eager"`, the Python attention forward the attention interior is read inside. The default is the checkpoint's, usually `sdpa`, and the interior is then unavailable ([Attention](../guide/values/attention.md#eager-attention)).

**Envoy**: nnsight's proxy for one module, reached by attribute path, with `.input`, `.output` and `.source` ([nnsight.net](https://nnsight.net/features/6_modules/)). nnterp's `Layer`, `Attention`, `Mlp` and the other hosts are `Envoy` subclasses installed through `envoys=`.

**EProperty**: the descriptor every value is: nnsight's `eproperty` with a key that is a path from the host (`"output"`, `"../ln_2.output"`, `"source.<op>.inputs"`) and an `unavailable=` reason ([Descriptors](../developing/internals/descriptors.md)).

**Expert**: one of a mixture's routed feed-forward networks, `num_experts` per block ([Mixture of experts](../guide/architectures/mixture-of-experts.md)).

**Family**: the module under `nnterp/families/` for one `model_type`, such as `nnterp.families.gpt2`, declaring `RENAME`, its `Layer`, `Attention` and `Mlp` subclasses and `ENVOYS`. `model.family` is the one a checkpoint resolved to ([Adding a family](../developing/adding-a-family.md)).

**Forward order**: the order in which the forward reaches the values' locations: the inputs, then each block's interior before its outputs, block 3 before block 5, `logits` last. `model.order(i)` gives it for block `i` ([Attention](../guide/values/attention.md#read-order)).

**Getter**: the function an `EProperty` decorates, the preprocess: what it returns is what a read serves ([Overriding a value](../developing/overriding-values.md#a-value-that-has-to-be-a-copy)).

**Grouped-query attention**: fewer key/value heads than query heads. `attention_keys` and `attention_values` are read before `repeat_kv`, so their head axis is `num_kv_heads` wide ([Attention](../guide/values/attention.md#grouped-query-attention)).

**Head outputs**: `attention_head_outputs`, each head's mix of the values before the output projection, `[batch, seq, num_heads, head_dim]` ([Attention](../guide/values/attention.md)).

**Host**: the envoy a value is declared on and its path is walked from: `Layer` for `layer_output`, `Attention` for the pattern, the root for `logits` ([Custom values](../developing/custom-values.md)).

**Hybrid**: a family whose blocks are of two kinds, `linear_attn` on most blocks and `self_attn` on a few (Qwen3-Next, Qwen3.5, OLMo-Hybrid, Kimi-Linear), or Nemotron-H's and Bamba's mixer, attention and MLP blocks. `support()` reads per block there ([Availability](../guide/values/availability.md#three-forms)).

**Image features**: `vision.image_features`, the tensor the wrapper scatters into the token embeddings at the image tokens, `[image_tokens, hidden_size]`, so that `layers[0].input[vision.image_token_mask] == vision.image_features` ([Vision-language](../guide/architectures/vision-language.md)).

**Image token mask**: `vision.image_token_mask`, `input_ids == image_token_id` as a `[batch, seq]` bool, read off the inputs and so first in a trace ([Vision-language](../guide/architectures/vision-language.md)).

**Interface**: transformers' shared attention call, `attention_interface_1` (`nnterp.components.INTERFACE`), which under eager attention is `eager_attention_forward`. The base `Attention` reads the interior on and inside it, and a family with its own arithmetic relocates the names ([transformers compatibility](../developing/transformers-compatibility.md#the-shared-interface)).

**Invoke**: nnsight's `tracer.invoke(prompt)`, one prompt batch inside a trace ([nnsight.net](https://nnsight.net/features/8_batching/)). A mixture's routing values serve each invoke its own rows, on vLLM an invoke is one prompt, and a trace with images carries one image-carrying invoke ([Mixture of experts](../guide/architectures/mixture-of-experts.md#ablating-and-rerouting)).

**Kernel**: the function a recurrent mixer's forward calls to run the sequence: `torch_chunk_gated_delta_rule` on a prompt and `torch_recurrent_gated_delta_rule` on a decode step for a gated DeltaNet, the scan and the state update for Mamba. Everything but `attention_output` is read at that call ([Gated DeltaNet](../guide/architectures/delta-net.md#kernels-and-route_kernels)).

**Latent attention**: DeepSeek-V2's and V3's attention, where queries and keys are `qk_head_dim` wide and values `head_dim`, the one case where the two sizes differ ([Attention](../guide/values/attention.md)).

**Layout**: the one axis layout a value has on every family, a named `jaxtyping` type such as `Residual` or `Pattern`, read as `value.layout` with its axes as `value.dims` ([Layouts](../guide/values/layouts.md)).

**Mixer**: the sublayer that mixes information across positions, `self_attn` or `linear_attn`. On the architecture pages it is short for recurrent mixer ([Cross-family sweep](../tutorials/cross-family-sweep.md)).

**Module boundary**: a value read where a module takes or returns a tensor, between the block's stream values: `attention_norm_output`, `layer_mid`, `mlp_norm_output` and `mlp_input` ([Residual stream](../guide/values/residual-stream.md#between-the-sublayers)).

**Name**: an entry in the vocabulary, the one set of module names every family answers to: `embed_tokens`, `layers[i]`, `layers[i].self_attn`, `layers[i].mlp`, `norm`, `lm_head`, with `linear_attn` on a hybrid and `vision` and `projector` on a wrapper. A name says which module; a value says which tensor ([Names](../guide/names.md)).

**Native name**: the module's name in the checkpoint's own architecture, `transformer.h[i].attn` on GPT-2. It keeps working, `envoy.path` is always the native path, and above the host an `EProperty` path takes native names only ([Names](../guide/names.md)).

**Occurrence**: nnsight's count of how many times a run has reached one location. `tracer.iter[t]` binds a read to occurrence `t`, which is how `state` is walked token by token ([Recurrent mixers](../developing/internals/recurrent-mixers.md#the-state-after-every-token)).

**Operation**: one call or binding inside a module's forward as nnsight's `.source` labels it, `{callable}_{occurrence}`: `nn_functional_softmax_0`, `attention_interface_1`. A `source.` path names one ([Finding source ops](../developing/finding-source-ops.md)).

**Parallel block**: a block where one norm feeds both sublayers and `x + attn(norm(x)) + mlp(norm(x))` is summed at the end (GPT-J, Phi, Falcon, Pythia). It has no `layer_mid`, and the contribution identity holds to rounding ([Availability](../guide/values/availability.md#the-reasons)).

**Pattern**: `attention_probabilities`, the distribution the values are mixed with, after the softmax and its dropout, `[batch, num_heads, query, key]`. `attention_scores` is the same tensor before the softmax ([Attention](../guide/values/attention.md)).

**Pinned read**: a read inside `tracer.iter[t]`, bound to occurrence `t` of its location. A read outside any loop is relaxed and takes the occurrence in flight ([Recurrent mixers](../developing/internals/recurrent-mixers.md#the-state-after-every-token)).

**Postprocess**: the hook that runs on an assigned value, and what it returns is written to the location. The base `layer_output` puts a tensor back into the block's tuple with it ([Overriding a value](../developing/overriding-values.md#a-value-that-has-to-be-a-copy)).

**Premix**: `attention_premix`, `o_proj.input` split by head: every head's output as the output projection takes it, served under any `attn_implementation` ([Residual stream](../guide/values/residual-stream.md#between-the-sublayers)).

**Projector**: `model.projector`, the last module before the scatter on a wrapper. `projector.input` is what the tower hands over, which is not `tower_output` on every host ([Vision-language](../guide/architectures/vision-language.md#the-projector)).

**Read order**: the order of the reads within one trace, which has to follow forward order. A read out of order raises nnsight's `OutOfOrderError` in a plain trace and binds the next step's occurrence under `tracer.iter` ([Generation](../guide/generation.md#a-read-out-of-order-inside-the-loop)).

**Reason**: the string `support()` gives for a value a checkpoint lacks and `Unavailable` carries, from the descriptor's `unavailable=`: a fixed string, or a predicate of the envoy evaluated on the instance ([Availability](../guide/values/availability.md#the-reasons)).

**Recurrent mixer**: a sublayer that mixes positions through a recurrent state rather than a pattern: a gated DeltaNet (`LinearAttention`), a Mamba-1 (`SelectiveScan`) or a Mamba-2 (`StateSpace`) mixer, all `RecurrentMixer` subclasses, served as `layers[i].linear_attn` under attention's names ([Gated DeltaNet](../guide/architectures/delta-net.md)).

**Registry**: `nnterp.families`: `lookup(model_type)` returns what `register` put in `REGISTRY`, else `nnterp.families.<model_type>` imported on first use, else the default family ([Registering](../developing/registering.md#the-lookup-order)).

**Residual stream**: the `[batch, seq, hidden_size]` tensor a block passes to the next, `layer_input` entering and `layer_output` leaving ([Residual stream](../guide/values/residual-stream.md)).

**Router**: a mixture's gate, `moe.router`, whose `router_logits` are scored to pick each token's experts ([Mixture of experts](../guide/architectures/mixture-of-experts.md)).

**Routing**: on a mixture, which experts a token goes to, the pair `expert_weights` and `expert_indices`, `[batch, seq, top_k]` ([Mixture of experts](../guide/architectures/mixture-of-experts.md)). The [Recurrent mixers](../developing/internals/recurrent-mixers.md#pure-torch-kernels-and-routing) page also says routing for `route_kernels`, the process-wide rebinding of a family's kernels to transformers' pure-torch functions.

**Sandwich block**: a block that norms a sublayer's output before adding it, `x + post_attention_layernorm(attn(input_layernorm(x)))` (Gemma-2, Gemma-3). The contribution is the post-norm's output, as it is on a post-norm block (OLMo-2, OLMo-3, EXAONE-4), which has no pre-norm ([Overriding a value](../developing/overriding-values.md#a-sibling-modules-output)).

**Scatter**: the step in a wrapper's forward that writes the projected image features into the token embeddings at the image tokens, a `masked_scatter`. `image_features` is read there ([Vision](../developing/internals/vision.md#where-image_features-is-read)).

**Size**: a read-only attribute read off the config: the root's `num_layers`, `hidden_size`, `vocab_size`, `num_heads`, `num_kv_heads`, `head_dim`, `qk_head_dim` and `intermediate_size`, each a `StandardizedProperty` that a family can define instead. A block's own sizes are on its `self_attn` and `mlp`, read off the module ([The whole model](../guide/values/whole-model.md#sizes)).

**Slot**: one of a token's `top_k` routing choices, the third axis of `expert_weights`, `expert_indices` and `expert_outputs` ([Mixture of experts](../guide/architectures/mixture-of-experts.md)).

**Softcap**: Gemma-2's `final_logit_softcapping`, `cap * tanh(logits / cap)` after `lm_head`. `logits` and `project_on_vocab` include it, and `lm_head_output` does not ([The whole model](../guide/values/whole-model.md#logits-and-lm_head_output)).

**Source**: nnsight's `.source`, the operations of a module's forward ([nnsight.net](https://nnsight.net/features/11_source/)). A source-located value has a `source` segment in its path and is read inside the forward, before the module's own output ([Finding source ops](../developing/finding-source-ops.md)).

**State**: a recurrent mixer's memory: `state_input` entering a call, `state_output` leaving it, `state` after one token and `states` after every token, `[batch, heads, key_dim, value_dim]` on a gated DeltaNet or Mamba-2 and `[batch, channels, state_dim]` on Mamba-1 ([Gated DeltaNet](../guide/architectures/delta-net.md#the-state-after-every-token)).

**Sublayer**: the attention (or recurrent mixer) and the MLP of a block, each adding one contribution. `self_attn.input` and `mlp.input` are what enters each, whatever norm produced it ([Names](../guide/names.md#the-norms)).

**Support**: `model.support()`, every value with `None` or a per-block `{layer: reason}`; `model.support(layer=i)`, one block, flat; `envoy.support()`, one envoy. Computed from the config and the module tree, nothing runs ([Availability](../guide/values/availability.md#three-forms)).

**Tower**: `model.vision`, the vision encoder of a wrapper, a `Vision` envoy with its own `layers`, sizes, `patch_embeddings` and `tower_output`. Its blocks' stream values have the `Patches` layout ([Vision-language](../guide/architectures/vision-language.md)).

**Transform**: the hook that runs on the served view after an in-place edit and returns what the model receives when the block moves on. Falcon's `mlp_output`, a copy, carries edits back with one ([Overriding a value](../developing/overriding-values.md#a-value-that-has-to-be-a-copy)).

**Tuple-returning block**: a block whose forward returns `(hidden_states, ...)` (GPT-J, GPT-Neo, BLOOM, MPT, Falcon), whose family sets `Layer.returns_tuple`. `layer_output` is the tensor either way, and `skip_layers` packs the replacement as the block would ([Residual stream](../guide/values/residual-stream.md#a-tensor-on-every-family)).

**Unavailable**: `nnterp.Unavailable`, a `RuntimeError` raised at the read or write of a value the checkpoint lacks, before the model runs, with the reason `support()` gives. `unavailable("...")` is the marker a family assigns in place of a value it does not have ([Availability](../guide/values/availability.md#nnterpunavailable-at-the-read)).

**UnsupportedFamily**: `nnterp.UnsupportedFamily`, a `ValueError` raised at load when no family covers the `model_type` and the default family cannot standardize the checkpoint ([Loading](../guide/loading.md#a-model_type-with-no-family)).

**Value**: a standard value: a tensor served by an `EProperty` on a host and meaning the same thing on every family, such as `layer_output` or `attention_probabilities`. It is listed in `print(model)`, reported by `support()`, and read, edited in place or assigned the way `.output` is ([Residual stream](../guide/values/residual-stream.md#reading-editing-in-place-assigning)).

**Wrapper**: the image-text-to-text class around a text model, such as `LlavaForConditionalGeneration`. Its family is the text model's, from `text_config.model_type`, and `model.vision` serves anything only when it is loaded with `task="image-text-to-text"` ([Vision-language](../guide/architectures/vision-language.md)).

Every symbol these terms name is on [API at a glance](api-at-a-glance.md).
