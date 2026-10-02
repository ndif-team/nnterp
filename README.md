# nnter

One module vocabulary across transformer architectures, on top of
[nnsight](https://github.com/ndif-team/nnsight).

```python
from nnter import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2")   # or a Llama, a Pythia, ...

with model.trace("The Eiffel Tower is in"):
    attn = model.layers[5].self_attn.attention_output.save()   # what attention adds
    resid = model.layers[5].layer_output.save()                # a tensor on every family
    logits = model.lm_head.output.save()
```

The same block runs unchanged on `meta-llama/Llama-3.1-8B`,
`EleutherAI/pythia-70m-deduped`, and every other registered family whose block 5 has an
attention module (on a hybrid such as Qwen3.5 three blocks in four have `linear_attn`
instead, so pick the block from the ones with `self_attn`; a pure state-space model such
as Mamba has none, and `layer_output` is what exists on every block). The families: GPT-2,
Llama, Llama 4 (text), GPT-NeoX, Mistral, Mixtral, MiniMax-M2, Qwen2, Qwen2-MoE, Qwen3, Qwen3-MoE, Gemma,
Gemma-2, Gemma-3 (text and multimodal checkpoints), Gemma-4 (text; E2B/E4B, 26B-A4B, 31B and the unified 12B), GPT-OSS, DeepSeek-V2, DeepSeek-V3, DeepSeek-V3.2, GLM-4.5/4.6,
GLM-4.7-Flash, GLM-5, DBRX, Phi, Phi-3,
OLMo, OLMo-2, OLMo-3, OLMoE, EXAONE 4.0, SmolLM3, StableLM, Cohere (Command-R), Cohere-2, Granite, GPT-J, GPT-Neo, CodeGen, GPT-NeoX-Japanese, BLOOM, MPT, Falcon (7B and 40B
layouts), OPT, XGLM, the gated DeltaNet hybrids Qwen3-Next, Qwen3.5,
Qwen3.5-MoE (text) and OLMo-Hybrid, the Mamba-1 state-space models Mamba,
Falcon-Mamba and Jamba (with attention and experts), and the Mamba-2 models Mamba-2,
Nemotron-H, Bamba and Falcon-H1, on transformers 5.17. The vocabulary is Llama's block names, with the containers
lifted out of the inner `.model`:

| standard name                              | GPT-2                     | Llama                   | GPT-NeoX                          |
| ------------------------------------------ | ------------------------- | ----------------------- | --------------------------------- |
| `model.embed_tokens`                       | `model.transformer.wte`   | `model.model.embed_tokens` | `model.gpt_neox.embed_in`      |
| `model.layers[i]`                          | `model.transformer.h[i]`  | `model.model.layers[i]` | `model.gpt_neox.layers[i]`        |
| `model.layers[i].self_attn`                | `...h[i].attn`            | same                    | `...layers[i].attention`          |
| `model.layers[i].mlp`                      | same                      | same                    | same                              |
| `model.layers[i].self_attn.input`          | what enters the attention: the normed stream, or the block input where there is no pre-norm (OLMo-2/3) |||
| `model.layers[i].mlp.input`                | what enters the MLP: the pre-MLP norm's output, whatever the family calls that norm |||
| `model.norm`                               | `model.transformer.ln_f`  | `model.model.norm`      | `model.gpt_neox.final_layer_norm` |
| `model.lm_head`                            | same                      | same                    | same (`embed_out` on old releases)|

The original names keep working: a standard name is an nnsight `rename` alias,
an extra attribute pointing at the same envoy.

Norm names are deliberately not part of the vocabulary. `input_layernorm` and
`post_attention_layernorm` bind as aliases where a family spells them
otherwise, but their *meaning* varies: on Gemma-2/3/4 `post_attention_layernorm`
follows the attention and the pre-MLP norm is `pre_feedforward_layernorm`; on
a parallel block (GPT-NeoX, Phi, GPT-J, StableLM, Falcon) one norm feeds both
sublayers. What a user wants from a norm is what it produces, and that is
`self_attn.input` and `mlp.input` on every family, checked per family by the
test suite against the family's own norm.

## Docs

`docs/` holds one page per feature (`usage/`), interpretability recipes written once against
the standard values (`patterns/`), how to add or override a family (`extending/`), the
internals (`developing/`) and the API and families tables (`reference/`). `CLAUDE.md` routes
a task to the right page. Every snippet in them has run against the pinned checkpoints.

## How it works

`StandardizedTransformer` subclasses `TransformersModel`. Its `__init__` reads
the checkpoint's config, looks up `config.model_type` in
`nnter.families.REGISTRY`, and passes that family's `RENAME` dict as nnsight's
`rename=`. A `rename=` of your own is merged on top.

Each family is one module under `nnter/families/` declaring `MODEL_TYPES`,
`RENAME`, its `Layer`, `Attention` and `Mlp` subclasses and `ENVOYS`, plus a
`def <size>(model)` for any root size its config spells its own way (Falcon's
`num_kv_heads`, GPT-2's `intermediate_size` from `n_inner`); the root's
`StandardizedProperty` calls it in place of the plain rule. To add one,
name the module after the model type (`gemma3_text.py` covers `gemma3_text`),
and that is the registry: a family's module is imported the first time a
checkpoint of that type is loaded, so `import nnter` imports no transformers
modeling module. A family from elsewhere, or an override of a shipped one,
goes through `nnter.families.register()`. Families whose checkpoints already use Llama's
names (Mistral, Qwen, ...) need only the three container keys, like `llama.py`.

A key with several components (`transformer.h`) binds where it resolves from,
the root, which is what lifts `layers` up; a single-component key (`attn`)
binds in every block that has one.

## Standard values

Beyond names, some modules get standard *values*. Every decoder block is
wrapped by the family's `Layer`, an nnsight `Envoy` subclass, and gains:

| value                       | what it is                                                                 |
| --------------------------- | -------------------------------------------------------------------------- |
| `model.layers[i].layer_output` | the residual stream leaving the block, as a tensor whether the block returns a tensor (Llama, GPT-2, GPT-NeoX) or a tuple with it first (GPT-J, GPT-Neo, Bloom, MPT, Falcon) |
| `model.layers[i].self_attn.attention_output` | what the attention sublayer adds to the residual stream |
| `model.layers[i].mlp.mlp_output` | what the MLP sublayer adds to the residual stream |
| `model.layers[i].self_attn.attention_probabilities` | the attention pattern the values are mixed with, `[batch, heads, query, key]`, read at the dropout after the softmax inside the eager attention forward |
| `model.layers[i].self_attn.attention_queries` / `attention_keys` / `attention_values` | what the attention interface receives: queries `[batch, heads, seq, qk_head_dim]` after RoPE, keys `[batch, kv_heads, seq, qk_head_dim]` and values `[batch, kv_heads, seq, head_dim]` before `repeat_kv` |
| `model.layers[i].self_attn.attention_scores` | the masked, scaled scores entering the softmax, `[batch, heads, query, key]`; `softmax(scores)` is the pattern up to the dtype cast |
| `model.layers[i].self_attn.attention_head_outputs` | each head's output before concatenation and the output projection, `[batch, seq, heads, head_dim]` |

The root answers for the whole model too:

| value                      | what it is                                                                  |
| -------------------------- | --------------------------------------------------------------------------- |
| `model.logits`             | the model's final logits, softcapping applied (Gemma-2); `lm_head.output` is the raw projection |
| `model.token_embeddings`   | the embedding module's output, with any scale the module applies (Gemma's); positional embeddings, embedding norms and multipliers the model applies after it (GPT-2's `wpe`, Granite's `embedding_multiplier`) are not in it, so `layers[0].input` is what enters block 0 |
| `model.next_token_probs`   | `logits[:, -1].softmax(-1)`, derived from the output; read-only, assign `logits` instead |
| `model.num_layers`, `num_heads`, `num_kv_heads`, `head_dim`, `qk_head_dim`, `hidden_size`, `intermediate_size`, `vocab_size` | sizes from the config: a plain rule at the root (`head_dim` is the config's where it says, as on Qwen3 and Gemma, else `hidden // heads`), and the family's own spelling where its config differs (Falcon's `num_kv_heads`, DeepSeek's `v_head_dim`, GPT-2's `n_inner`) |

The two contributions are defined by the identity
`layers[i].input + attention_output + mlp_output == layers[i].layer_output`,
which holds for a sequential and a parallel block alike; the test suite checks
it on every pinned family. A family whose sublayer adds the residual inside the
module (BLOOM, MPT, DBRX) or adds a post-sublayer norm's output instead
(Gemma-2/3/4, OLMo-2) points the value at the right place in its subclass, so the
name means the same thing everywhere. The pattern is read after the dropout, not
at the softmax, for the same reason: that is the tensor the values are mixed
with, in the model's dtype and with an attention sink's column already dropped.

Read them, edit them in place, or assign to them; a tuple module gets its other
elements back unchanged. Assigning one argument of the attention call replaces
just that argument. Two torch limits on in-place edits: GPT-2's queries, keys
and values are split views of one tensor, which torch refuses to edit in place
(assign instead), and Falcon's `mlp_output` is a copy, carried back into the
model by an `eproperty` transform.

The interior values live on transformers' shared attention interface on most
families; GPT-J, GPT-Neo, CodeGen, GPT-NeoX-Japanese, XGLM, BLOOM, MPT and Falcon do their own attention arithmetic, and
their families map the same five values onto their own operations (with the
head outputs presented sequence-first as a view where the family keeps heads
first). Two of them bind values at points a single trace must respect: on
Falcon the values bind before the rotary embedding that produces the queries
and keys, so read them first. Two families change what the interior means:
GPT-OSS's attention sink makes the pattern's rows sum to less than one and
puts `attention_scores` at the masked scores just before the sink column joins
them; DeepSeek's latent attention gives queries and keys `qk_head_dim` and the
interface `num_heads` key/value heads, so the root also publishes
`qk_head_dim`.

Both are nnsight `eproperty` descriptors, so they show up in the model's repr
with their description. `attention_probabilities` reaches into the forward
through nnsight's `.source`, so it needs the eager attention path: load with
`attn_implementation="eager"`, or the value is unavailable (`support()` says so,
and a read raises `Unavailable` naming the implementation the model runs).

Its key is a path into the forward,
`"source.attention_interface_1.source.nn_functional_dropout_0.output"`. An
operation inside a called function only exists once someone has drilled into
that call in the *current* run (the interleaver resolves the callee from the
live value and clears what it built at the start of every trace), so the
descriptor walks the path before every read or write, then serves the
location as an ordinary eproperty.

`nnter.components` holds `Layer`, `Attention`, `Mlp` and one descriptor,
`EProperty`, whose key is a path from the host envoy: `"output"` for the
host's own output, `"../post_attention_layernorm.output"` or
`"embed_tokens.output"` for a value produced by another module named relative
to this one (a sandwich block's post-sublayer norm, the root's embedding), and
`"source.<op>.output"` for an operation inside the forward. Each family subclasses the three envoys,
overriding only what its forward spells differently, and keys them on its own
transformers module types in its `ENVOYS` (`envoys=` matches by type or native
path, never by alias). Three shapes of override exist today:

- **sandwich norms** (Gemma-2/3/4, OLMo-2/3, EXAONE-4): the contributions are the
  post-attention and post-feedforward norms' outputs, via a `../` path to the
  sibling norm (Gemma-4's block then adds a third term, `layers[i].per_layer_output`,
  on checkpoints with per-layer embeddings, and multiplies the sum by `layer_scalar`);
- **residual added inside the module** (BLOOM both sublayers, MPT's MLP): the
  contribution is the operation before the add, via a `source.` path, reading
  `dropout_add`'s first argument or the dropout's output;
- **own attention arithmetic** (GPT-J, GPT-Neo, CodeGen, GPT-NeoX-Japanese, XGLM, BLOOM, MPT, Falcon): the pattern and the
  interior values are that family's own operations rather than the shared
  interface. A family that has not mapped them yet marks them
  `unavailable(NOT_ON_INTERFACE)`.

Falcon's block adds the attention into the MLP's output tensor in place, so its
`mlp_output` reads a copy; assign to edit it. DeepSeek-V4 carries several parallel
residual streams: `layer_output` is the block's own `[batch, seq, streams, hidden]`,
the contributions are the sublayers' own outputs, and the block's stream weights are
four values on its `Layer` (docs/reference/families.md, "Hyper-connection residual"). OPT has no MLP module, and
`support()` lists no `mlp` value for it.

Pass `envoys=` to `StandardizedTransformer` to add your own; yours replace the
family's on the same key. nnsight tries type keys before path keys, so to
displace a family's type-keyed envoy, key yours on the type too. A value on a
class installed this way is listed by `model.support()` like the family's own.

## Hybrids: gated DeltaNet

Qwen3-Next and Qwen3.5/3.6 replace three blocks in four with a gated DeltaNet
mixer, `linear_attn`. It projects queries, keys and values like attention but
mixes them through a per-head recurrent state: each token decays the state by
a learned gate, writes its key/value pair in scaled by a beta, and the query
reads against it. There is no pattern and no scores. `nnter.components.LinearAttention`
gives such a block:

| value | what it is |
| --- | --- |
| `layers[i].linear_attn.attention_output` | what the mixer adds to the residual stream |
| `attention_queries` / `attention_keys` / `attention_values` | what the delta rule receives, `[batch, seq, heads, dim]` |
| `decays` | the per-token log decay of the state, `[batch, seq, heads]`, float32, non-positive |
| `betas` | the per-token write strength, `[batch, seq, heads]`, in (0, 1) |
| `state_input` / `state_output` | the recurrent state entering and leaving the layer, `[batch, heads, key_dim, value_dim]` (`None` entering on a fresh prompt) |
| `attention_head_outputs` | each head's read of the state, `[batch, seq, heads, value_dim]` |

A block has either `self_attn` or `linear_attn`, so `support()` reports each
`self_attn` value as missing on the linear blocks and vice versa, per block.
The values are read at the delta-rule kernel call. A prompt runs the chunked
kernel and each decode step of `generate` the recurrent one, two different
operations in the forward; the forward binds `use_precomputed_states` before
it branches, and the values read that binding and the call's length to name
the call that fires on this step (`RecurrentMixer.KERNEL`), so the same value
works in a `trace` and at every step of `tracer.iter`, and the state hands
off from one step to the next. They need transformers' pure-torch kernels: with
`flash-linear-attention` or `causal-conv1d` installed the kernel has no Python
source, and `support()` says so.

The state *after every token* of a prompt is a further step, and like eager
attention it is a choice made at load. The chunked kernel a prompt normally
runs through carries the state between 64-token chunks and never materializes
it per token; transformers' token-by-token kernel does, at a cost.
`nnter.route_kernels(model.family, "torch")` routes the
family's prompts through it (process-wide, like installing a kernel; call it
before tracing a layer; `"default"` restores the default), and then:

```python
model = StandardizedTransformer("Qwen/Qwen3.5-9B", attn_implementation="eager")
route_kernels(model.family, "torch")          # before the first trace of a linear block
mix = model.layers[0].linear_attn

per_token = []
with model.trace(prompt) as tracer:
    for t in tracer.iter[:]:                  # `state` is a per-token location: nnsight's own iteration walks it
        per_token.append(mix.state.save())

with model.trace(prompt) as tracer:
    for t in tracer.iter[6]:
        s6 = mix.state.save()                 # the state after token 6
    for t in tracer.iter[7]:
        mix.state = torch.zeros_like(s6)      # a write at token 7: the tokens after it continue from zeros
    for t in tracer.iter[8]:                  # positions inside the prompt: a later one is never reached
        s8 = mix.state.save()

with model.trace(prompt):
    states = mix.states.save()                # every position stacked, [batch, seq, heads, key_dim, value_dim]
    # mix.state_after(7) and mix.set_state_after(7, value) are the two iter forms above, as calls
```

Under `generate` the prompt is one kernel call and each decode step another,
over one token, so the per-token view is an inner loop on step 0 and then one
state per step:

```python
with model.generate(prompt, max_new_tokens=5) as tracer:
    for step in tracer.iter[:]:
        if step == 0:
            for t in tracer.iter[:n]:         # the prompt's tokens
                s = mix.state.save()
        else:
            s = mix.state_output.save()       # the one token this step processes
```

`states`, `state_after` and `set_state_after` count from the current call's
own first token on every step (occurrences are per location over the whole
run, so they offset by what earlier steps put through the op).

Without the switch, reading any of them raises `Unavailable` with that
instruction, and `support()` reports it. The results are the same to float error: the two kernels compute
the same rule. Reads follow the forward: in one trace, positions before a
write come before it and positions after it come after; `states` reads every
position, so it goes in a trace of its own.

## Layouts

Every value has one layout on every family, and says what it is: each is
annotated with one of thirty-one named `jaxtyping` types defined beside the
envoy that serves it, `Residual = Float[Tensor, "batch seq hidden"]`,
which `value.layout` returns (`Layer.layer_output.layout is Residual`) and
`value.dims` names (`("batch", "seq", "hidden")`).
`isinstance(tensor, Attention.attention_queries.layout)`, or
`isinstance(tensor, Queries)`, checks rank and dtype, and the suite checks
every value's axes against the model's sizes on every family. A family that
redefines a value annotates it with the same name, so it cannot drift from
the base; a value of your own does the same (`from nnter.components import
Residual`; the root's `Logits`, `NextTokenProbs` and `Tokens` come from
`nnter.standardized`). Layouts differ between values, not between families,
with one exception: `layer_output` is `Streams` on DeepSeek-V4, whose residual is
several parallel streams. The main ones (the Mamba and mixture-of-experts layouts are in
`docs/usage/layouts.md`):

| layout | axes | values |
| --- | --- | --- |
| `Residual` | `batch seq hidden` | `layer_output`, `attention_output`, `mlp_output`, `token_embeddings`, `self_attn.input`, `mlp.input` |
| `Streams` | `batch seq streams hidden` | `layer_output` and `layers[i].input` on DeepSeek-V4 |
| `StreamWeights` / `StreamMixing` | `batch seq streams` / `batch seq streams streams` | DeepSeek-V4's `attention_post`, `mlp_post` / `attention_comb`, `mlp_comb` |
| `Logits` / `NextTokenProbs` | `batch seq vocab` / `batch vocab` | `logits` / `next_token_probs` |
| `Tokens` | `batch seq` (`Int`) | `input_ids`, `attention_mask` |
| `Queries` | `batch heads seq qk_head_dim` | `attention_queries` |
| `Keys` / `Values` | `batch kv_heads seq qk_head_dim` / `batch kv_heads seq head_dim` | `attention_keys` / `attention_values` |
| `Pattern` | `batch heads query key` | `attention_scores`, `attention_probabilities` |
| `HeadOutputs` | `batch seq heads head_dim` | `attention_head_outputs` |
| `LinearQK` / `LinearV` | `batch seq heads key_dim` / `batch seq heads value_dim` | `linear_attn.attention_queries`, `keys` / `values`, `attention_head_outputs` |
| `Gates` | `batch seq heads` | `linear_attn.decays`, `betas` |
| `State` | `batch heads key_dim value_dim` | `state_input`, `state_output`, `state` |
| `States` | `batch seq heads key_dim value_dim` | `states` |

Batch is axis 0 everywhere. The sequence axis is 1 on every value that has
one except softmax attention's queries, keys and values, where it is 2, the
layout transformers hands its attention interface.

## Doing things with the values

Five methods on the model do the common things, written once against the
standard names so they run on every family:

```python
with model.trace(prompt):
    model.skip_layers(4, 7)                       # blocks 4..7 do not run; the stream passes straight through
    resid = model.layers[8].layer_output.save()
    lens = model.project_on_vocab(resid).save()   # logit lens at block 8
    model.steer(10, vector, factor=3, token_positions=-1)   # add to the residual stream leaving block 10
    logits = model.logits.save()

model.get_topk_closest_tokens(resid[0, -1], k=5)   # [{token: probability}] for that position, projected the same way
model.probs_to_dict(logits[0, -1].softmax(-1), k=5)   # a distribution as {token: probability}
```

`skip_layers` hands each skipped block's input on as its `layer_output`,
packed the way that family's block would have returned it (`Layer.skip_with`;
a tuple family declares `returns_tuple`). `steer` adds in place, so it reaches
the model. `project_on_vocab` applies the final norm, `lm_head` and the
model's softcapping, so at the last block it equals `logits`.

## Inputs, tokenizers, prompts and activations

Inside a trace the model answers for its input: `model.input_ids` and
`model.attention_mask` (assignable: the model then runs on what you set) and
`model.input_size`, all values that print with the model. `tokenizer_kwargs=` at load
sets attributes on the tokenizer (`padding_side="left"`, a `pad_token`), and
`model.add_prefix_false_tokenizer` is the checkpoint's tokenizer with
`add_prefix_space=False`, so `"word"` and `" word"` are different tokens.

Two helper modules carry nnterp's names, written against the standard values:

- `nnter.prompt_utils`: `get_first_tokens(words, model)`, `Prompt.from_strings(text, targets, model)`
  with `has_no_collisions` and `get_target_probs`, and `run_prompts(model, prompts, batch_size)`
  returning each target's probability mass per prompt.
- `nnter.nnsight_utils`: `get_token_activations(model, prompts, layers, idx)` giving
  `[num_layers, num_prompts, hidden]` at one position of every prompt (default the last, which
  needs left padding), `collect_token_activations_batched` and
  `collect_last_token_activations_session` over many prompts, and `compute_next_token_probs`.

## What a checkpoint has

Not every checkpoint has every value: OPT has no MLP module, a model loaded
with sdpa cannot expose the eager pattern, a GPT-2 checkpoint with
`reorder_and_upcast_attn` leaves the shared attention path, a hybrid's
linear-attention blocks have no softmax. nnter
answers that before any trace runs:

```python
model = StandardizedTransformer("facebook/opt-125m")
model.support()
# {'logits': None, 'token_embeddings': None, 'next_token_probs': None,
#  'layer_output': None,
#  'self_attn.attention_output': None,
#  'self_attn.attention_probabilities': {0: "read inside the eager attention forward, but this model runs 'sdpa'; ...", ...},
#  ...}                       # no 'mlp.*' key: no block has an mlp module
model.support(layer=3)        # one block, flat
model.layers[3].self_attn.support()   # one envoy
```

`None` means available; otherwise the reason, per block where it differs. The
keys come from the tree: the root's values and every standard module on each
block, so a module no block has (OPT's `mlp`) is not listed, and one some
blocks lack (a hybrid's `self_attn`) reads `no self_attn module on this block`
there.
Reading an unavailable value raises `nnter.Unavailable` with the same reason,
at that line, before the model runs.

Every nnter descriptor takes `unavailable=`: a reason string, or a function of
the envoy returning one or `None`, evaluated on the instance so the config can
decide (`components.needs_eager` is the one the pattern uses). A family that lacks a
value altogether assigns `attention_probabilities = unavailable("...")` in its
class body; the name stays in the tree and the repr shows the reason. A module
some blocks lack (a hybrid's `self_attn`) is reported missing there from the
tree, with nothing to declare; one no block has (OPT's `mlp`) is not listed.

## Remote (NDIF)

`remote=True` works the way it does for any `TransformersModel`: the model's
remote key names `TransformersModel`, so a model the server deploys plain is
the one a `StandardizedTransformer` reaches. The block is re-run on the server
against the client's envoy tree, which carries the aliases and the family's
envoy classes by reference, so the server needs nnter installed at the same
version. Do not ship nnter by value (`nnsight.register`): an installed package
is pickled by reference anyway, and an `eproperty` cannot be pickled by value.

## transformers version

Developed against transformers 5.17. The operation names inside a forward are
what releases rename (the interface call, the dropout op, DBRX moving onto the
shared interface), and the per-family suite is the guard: every source-located
value must resolve on every layer and a written pattern must move the logits.

## Import order

Import nnsight (or nnter) before `transformers.modeling_layers`. On this stack
the reverse order segfaults at import; a plain `import transformers` first is fine.

## Tests

```
HF_HUB_OFFLINE=1 pytest
```

One file per family under `tests/families/` (92 families, 96 checkpoints), each subclassing `FamilySuite`
(`tests/families/suite.py`) with its pinned tiny checkpoint, native paths and
quirks, plus the tests that are specific to it. The suite is every end-to-end
statement a family must satisfy: aliases reach the native modules; every
standard value reads, writes, and appears in `support()` exactly as the family
expects; the contribution identity holds; a standardized model's activations
and logits equal a raw one's; the pattern's shape and row sums; every
source-located value resolves on every layer and a written pattern moves the
logits (an address can read a perfectly good tensor nothing downstream uses);
the interior's shapes, causal writes and in-place edits; the root values and
sizes against real tensors. `tests/` root holds what is not about one family:
the registry and load path, and the base envoys and descriptors.

Adding a family means adding one test file; adding a value means adding one
method to the suite.

## Typing

`model.layers` is annotated `Sequence[Layer]` and a `Layer`'s `self_attn`,
`linear_attn` and `mlp` as `Attention`, `LinearAttention` and `Mlp`, so an
editor completes `model.layers[3].self_attn.attention_probabilities`. The
annotations describe the family's subclasses; at runtime the objects are those
subclasses, reached through nnsight aliases.
