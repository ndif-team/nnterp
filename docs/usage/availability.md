---
title: Availability
one_liner: "`model.support()` says which standard values this checkpoint has and why not, before any trace; reading an unavailable one raises `nnterp.Unavailable` at that line."
tags: [usage, support, Unavailable, SourceNotAvailable, eager, hybrids]
related: [docs/usage/loading.md, docs/usage/vocabulary.md, docs/usage/residual-stream.md, docs/usage/layouts.md]
sources: [nnterp/standardized.py, nnterp/order.py, nnterp/components/standard.py, nnterp/components/eproperty.py, nnterp/components/attention.py, nnterp/components/linear_attention.py, nnterp/components/recurrent.py, nnterp/families/gpt2.py, nnterp/families/falcon.py, nnterp/families/opt.py]
---

# Availability

## What this is for

Not every checkpoint has every standard value. OPT has no MLP module; a model loaded with
`sdpa` never builds the attention pattern; a GPT-2 checkpoint with `reorder_and_upcast_attn`
leaves the shared attention path; a hybrid's linear blocks have no softmax and its per-token
state needs a kernel switch. Every nnterp value can say when it is not there and why, and `support()`
collects those reasons from the config alone, so a script can decide what to read before
running anything.

`None` means available. Anything else is the reason, a string.

## Canonical pattern

```python
from nnterp import StandardizedTransformer

model = StandardizedTransformer("facebook/opt-125m", dispatch=True)

support = model.support()
support["layer_output"]                       # None: available on every block
support["self_attn.attention_probabilities"]  # {0: "read inside the eager attention forward, but this model runs 'sdpa'; ...", ...}
"mlp.mlp_output" in support                   # False: no block has an mlp module, so there is no key

if support["self_attn.attention_probabilities"] is None:
    with model.trace(prompt):
        pattern = model.layers[3].self_attn.attention_probabilities.save()
```

## The three forms

`model.support()` is the root's values plus every block value, by dotted name. A block value
is `None` when every block has it, else `{layer: reason}` for the blocks that do not, so a
hybrid reads as a short dict. `model.support(layer=i)` is one block, flat.
`envoy.support()` is one envoy's own values.

The keys come from the tree. Every child of a block that carries standard values (a
`Standard` envoy) is walked under its standard name, so a value added through `envoys=`
is listed as `self_attn.<name>` or `mlp.<name>`, exactly as in that envoy's own
`support()`; a module some blocks lack (a hybrid's `self_attn`) is reported as
`no self_attn module on this block` on those; a module no block has (OPT's `mlp`) has no
key at all.

OPT, loaded without `attn_implementation`, trimmed to the interesting keys:

```python
>>> model.support()
{'logits': None, 'token_embeddings': None, 'next_token_probs': None,
 'input_ids': None, 'attention_mask': None, 'input_size': None,
 'layer_output': None,
 'self_attn.attention_output': None,
 'self_attn.attention_probabilities': {0: "read inside the eager attention forward, but this model runs 'sdpa'; load with attn_implementation='eager'",
                                       1: "read inside the eager attention forward, but this model runs 'sdpa'; load with attn_implementation='eager'",
                                       ...},
 'self_attn.attention_queries': {0: "read inside the eager attention forward, but this model runs 'sdpa'; load with attn_implementation='eager'", ...},
 ...}  # attention_keys, attention_values, attention_scores, attention_head_outputs: the same reason; no 'mlp.mlp_output' key

>>> model.support(layer=0)
{'layer_output': None,
 'self_attn.attention_output': None,
 'self_attn.attention_probabilities': "read inside the eager attention forward, but this model runs 'sdpa'; load with attn_implementation='eager'",
 ...}

>>> model.layers[0].self_attn.support()
{'attention_queries': "read inside the eager attention forward, but this model runs 'sdpa'; load with attn_implementation='eager'",
 ...
 'attention_output': None,
 'attention_probabilities': "read inside the eager attention forward, but this model runs 'sdpa'; load with attn_implementation='eager'"}
```

Loaded with `attn_implementation="eager"`, every entry is `None`; `mlp.mlp_output` is
still absent, since no block has the module.

A hybrid (`"Qwen/Qwen3.5-9B"`, `attn_implementation="eager"`; on the tiny checkpoint
blocks 0-2 are linear and block 3 is attention):

```python
>>> model.support()
{...,
 'self_attn.attention_output': {0: 'no self_attn module on this block', 1: 'no self_attn module on this block', 2: 'no self_attn module on this block'},
 'self_attn.attention_probabilities': {0: 'no self_attn module on this block', 1: ..., 2: ...},
 'linear_attn.attention_output': {3: 'no linear_attn module on this block'},
 'linear_attn.decays': {3: 'no linear_attn module on this block'},
 'linear_attn.state_output': {3: 'no linear_attn module on this block'},
 'linear_attn.state': {0: "the state after each token is materialized only by the token-by-token kernel; the chunked kernel a prompt runs through carries it between chunks. Call nnterp.route_kernels(model.family, 'torch') before tracing this layer (slower, like attn_implementation='eager')",
                       1: ..., 2: ...,
                       3: 'no linear_attn module on this block'},
 'linear_attn.states': {0: "the state after each token is materialized only by ...", 1: ..., 2: ..., 3: 'no linear_attn module on this block'},
 'mlp.mlp_output': None}
```

A block has either `self_attn` or `linear_attn`, so each side's values are reported
missing on the other's blocks, per block. `attention_output`, `attention_queries`,
`attention_keys`, `attention_values` and `attention_head_outputs` exist under both names
and mean the same thing; `attention_scores` and `attention_probabilities` exist only on
`self_attn`, `decays`, `betas` and the state values only on `linear_attn`.

## Reading an unavailable value

Reading or writing one raises `nnterp.Unavailable` with the same reason, at that line,
before the model runs:

```python
with model.trace(prompt):
    pattern = model.layers[0].self_attn.attention_probabilities.save()
# Unavailable: model.model.decoder.layers.0.self_attn.attention_probabilities is not available:
#   read inside the eager attention forward, but this model runs 'sdpa'; load with attn_implementation='eager'
```

A module that does not exist on a block is an ordinary missing attribute: `model.layers[0].mlp`
on OPT raises `AttributeError`, and `getattr(layer, "mlp", None)` is `None`. Use that form
outside the trace to pick blocks.

## The reasons you will see

| reason (exact prefix) | value(s) | when |
| --- | --- | --- |
| `read inside the eager attention forward, but this model runs 'sdpa'; load with attn_implementation='eager'` | the attention interior on every interface family | loaded with any `attn_implementation` other than `"eager"` |
| `this checkpoint sets reorder_and_upcast_attn, which takes GPT-2's own upcast attention path` | the GPT-2 interior | `config.reorder_and_upcast_attn` is set |
| `The attention does its own arithmetic rather than transformers' shared attention interface; not mapped for this family yet` | an interior value a family has not mapped onto its own arithmetic | a family off the shared interface that marks it `unavailable(NOT_ON_INTERFACE)` |
| `no self_attn module on this block` / `no linear_attn module on this block` / `no mlp module on this block` | every value of that module | some other block has the module and this one does not (a hybrid's blocks); a module no block has (OPT's `mlp`) has no key instead |
| `the state after each token is materialized only by the token-by-token kernel; the chunked kernel a prompt runs through carries it between chunks. Call nnterp.route_kernels(model.family, 'torch') before tracing this layer (slower, like attn_implementation='eager')` | `linear_attn.state`, `linear_attn.states` | the family's delta rule is still the chunked kernel |
| `this checkpoint has no per-layer embeddings (hidden_size_per_layer_input is 0); the block adds attention_output and mlp_output only` | `per_layer_output` (Gemma-4's `Layer` only) | a Gemma-4 checkpoint without per-layer embeddings (26B-A4B, 31B) |
| `this mixer's kernels do not materialize the state per token` | `state`, `states` on a `RecurrentMixer` with no `STATE_OP` | the mixer has no token-by-token kernel |
| `the default family cannot tell what this sublayer adds to the stream; add a family module (docs/extending/adding-a-family.md)` | `attention_output`, `mlp_output` | the default family (a `model_type` with no family) serves no contribution ([loading](loading.md#an-architecture-with-no-family)); `project_on_vocab` and `get_topk_closest_tokens` raise the same way, "the default family cannot tell what follows lm_head" |
| `no self_attn module found under the names the default family knows; ...` / `no mlp module found ...` | every value of that module | the default family found no such module on the block |
| `read inside transformers' pure-torch torch_chunk_gated_delta_rule, but this process dispatches it to an optimized kernel (fla) with no Python source; uninstall it, or call nnterp.route_kernels(model.family, 'torch'), to read these` | every `linear_attn` value but `attention_output` | `flash-linear-attention` or `causal-conv1d` is installed |
| `a text-only load: no processor, so no image reaches the model; load with task='image-text-to-text'` | every value of `model.vision` and its blocks | a vision-language wrapper loaded under the default `task="text-generation"` ([vision.md](vision.md#availability)); `model.vision.support()` is then empty and `model.support()` has no `vision.` rows |
| `<tower>.image_size is not available: the tower takes images of any resolution, each cut into its own patch grid by the processor; read the grid off the processor's output (image_grid_thw, image_sizes, image_position_ids)` | `vision.image_size` (a size, so raised at the read; not a `support()` row) | the Qwen ViT, Pixtral, Gemma 4 and Gemma 4 unified |
| `the <family> family keys no ImageScatter on the '<model_type>' wrapper, so where its image features enter the text stream is unknown` | `vision.image_features` | the tower's names bind but the family does not say where the wrapper writes the features into the token embeddings |
| `the config names no image_token_id` | `vision.image_token_mask`, `vision.image_features` | a wrapper whose config has no image token id to read the mask from |

BLOOM and MPT do their attention arithmetic themselves, so their pattern and interior do
not need an eager load and are `None` under any implementation.

The reasons are evaluated on the instance, so a config that changes after the load changes
the answer: `route_kernels(model.family, "torch")` turns the `state` and `states`
entries to `None` on the linear blocks.

## `SourceNotAvailable`: the forward took another path

`support()` predicts from the config. A source-located value is then read at a named
operation inside the forward, and if this run's forward does not contain that operation
(a code path the family does not expect), the read raises nnsight's `SourceNotAvailable`
naming the value, the operation and what the forward does have:

```
SourceNotAvailable: model.transformer.h.0.attn.<value> reads operation '<op>' under .source, which
this run does not have: 'model.transformer.h.0.attn.source' has no operation '<op>'; available:
is_cross_attention_0, ..., attention_interface_0, attention_interface_1, .... The forward took a
path this family's toolkit does not expect.
```

`Unavailable` is the expected answer for a known configuration; `SourceNotAvailable` means
a forward the family has not been checked against (another transformers release renaming
an operation, a branch the config does not announce). See nnsight docs/usage/source.md for
how operations are named.

## Guarding code

Check `support()` outside the trace and branch there; the trace body then reads only what
exists:

```python
model = StandardizedTransformer(repo, dispatch=True, attn_implementation="eager")
support = model.support()
has_pattern = support["self_attn.attention_probabilities"] is None
attn_blocks = [i for i, layer in enumerate(model.layers) if getattr(layer, "self_attn", None) is not None]
has_mlp = support.get("mlp.mlp_output", "absent") is None   # .get: OPT has no key at all

with model.trace(prompt):
    for i in attn_blocks:
        if has_pattern:
            patterns[i] = model.layers[i].self_attn.attention_probabilities.save()
    if has_mlp:
        mlp = model.layers[0].mlp.mlp_output.save()
```

For one block, `model.support(layer=i)["self_attn.attention_probabilities"]` is the flat
form of the same check.

A vision-language wrapper adds its tower's rows under `vision.` (`"vision.image_features"`,
`"vision.self_attn.attention_probabilities"`), and `model.vision.support()` lists them
without the prefix; a text-only checkpoint has no `model.vision` attribute at all, so guard
on `getattr(model, "vision", None)` first and on `support()` second
([vision.md](vision.md#availability)).

## Forward order: `order` and `rank`

Within one invoke, values are read in the order the forward reaches them. `model.order()`
says that order for the values `support()` lists as available, and `model.rank()` gives one
sortable tuple per value. Like `support()`, both are asked outside a trace.

```python
from nnterp import StandardizedTransformer

model = StandardizedTransformer("hf-internal-testing/tiny-random-LlamaForCausalLM", attn_implementation="eager")

model.order(0)
# {'layer_input': 0, 'self_attn.attention_queries': 1, 'self_attn.attention_keys': 1,
#  'self_attn.attention_values': 1, 'self_attn.attention_scores': 2,
#  'self_attn.attention_probabilities': 3, 'self_attn.attention_head_outputs': 4,
#  'self_attn.attention_output': 5, 'mlp.mlp_output': 6, 'layer_output': 7}
model.order()
# {'input_ids': 0, 'attention_mask': 0, 'input_size': 0, 'token_embeddings': 1,
#  'logits': 2, 'next_token_probs': 2}

model.rank("input_ids")                            # (-1, 0): before the blocks
model.rank("self_attn.attention_queries", 1)       # (1, 1)
model.rank("logits")                               # (2, 2): num_layers, after the blocks

reads = [("logits", None), ("layer_output", 1), ("mlp.mlp_output", 0), ("token_embeddings", None)]
reads.sort(key=lambda read: model.rank(*read))     # token_embeddings, mlp_output@0, layer_output@1, logits
```

- **A rank is `(side, r)`.** `side` is `-1` for a root value the model serves before the
  blocks, the layer for a block value (named as in `support(layer=i)`), and `num_layers`
  for a root value after the blocks; `r` is the value's place in `order(side's layer)`.
  Sorting reads by rank sorts them into forward order, layer by layer.
- **Equal ranks are one location.** The queries, keys and values are the arguments of one
  attention call, so either can be read first. On Falcon the values come before the
  queries and keys (`rank("self_attn.attention_values", 0) < rank("self_attn.attention_queries", 0)`);
  on a Qwen3.5 linear-attention block the kernel's inputs (queries, keys, values, `betas`,
  `decays`, `state_input`) share one rank.
- **Per block shape.** A hybrid's blocks differ: `order(0)` on Qwen3.5 lists the
  `linear_attn.*` values and `order(3)` the `self_attn.*` ones.
- **Unavailable values are not ranked.** Loaded with `sdpa`, the scores and the pattern
  are absent from `order(i)`, and `rank` raises `Unavailable` with `support()`'s reason;
  a name that is no value at all raises `KeyError`.
- **Measured, then kept.** The first call runs one probe `model.scan` (meta tensors, no
  weights or dispatch needed; `nnterp.order.probe` says how). A forward that cannot run on
  meta tensors (Granite, OPT, a grouped-mm mixture of experts in float32) is probed with a
  `trace` instead, which loads the weights. The result is measured once and kept on the
  model, so set `nnterp.route_kernels` and `nnterp.chunk_per_token` before the model is
  dispatched, traced or measured: the same rule kernel routing already has for traces
  (`sourced` envoys instrument at dispatch). With torch kernels routed, a DeltaNet block
  also ranks `state` and `states`.
- **One forward pass.** The order is within one call of the model. Under `generate`, every
  step repeats it; `tracer.iter` picks the step. The vision tower's values
  (`vision.*`) are not ranked, and `StandardizedVLLM` raises `NotImplementedError`.

## Gotchas

- **`hasattr(envoy, "attention_probabilities")` raises.** Python's `hasattr` treats only
  `AttributeError` as absence; an unavailable value raises `Unavailable` through it, and an
  *available* source-located value read outside a trace raises `ValueError` (`Cannot access
  ... outside of interleaving`). This is a documented limitation; use `support()`.
- **`getattr(envoy, name, None)` inside a trace can trip a served value.** Decide which
  blocks have `self_attn` or `mlp` before the trace.
- **`support()` predicts; the forward decides.** A `SourceNotAvailable` at read time means
  the forward differs from what the family expects, not that the value is unavailable by
  configuration.
- **`attn_implementation` is transformers' default unless you pass it.** The most common
  reason in `support()` is the `'sdpa'` one, and the fix is in the message.
- **The repr lists a value whether or not this checkpoint has it.** Only a value a family
  marks `unavailable("...")` in its class body prints as `Unavailable: <reason>`; a
  config-dependent reason (eager, `reorder_and_upcast_attn`) shows only in `support()`.
- **A hybrid's `support()` is a long dict by design.** Each value is reported per block on
  the blocks that lack its module; read `support(layer=i)` for one block.

## Related

- [loading](loading.md): `attn_implementation="eager"`.
- [vocabulary](vocabulary.md): blocks without a module (OPT, hybrids).
- [residual-stream](residual-stream.md): the values that never need eager.
- [layouts](layouts.md): shapes, once a value is available.
- nnsight docs/usage/source.md: operation names and `SourceNotAvailable`.
