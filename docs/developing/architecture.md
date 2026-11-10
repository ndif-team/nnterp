---
title: Architecture
one_liner: The map of nnterp — how a checkpoint's model_type becomes a renamed envoy tree whose blocks carry standard values, and which layer owns what.
tags: [developing, architecture, internals, families, components]
related: [docs/developing/eproperty-internals.md, docs/developing/recurrent-mixer-internals.md, docs/developing/testing.md, docs/developing/gotchas.md, docs/extending/index.md]
sources: [nnterp/standardized.py, nnterp/families/__init__.py, nnterp/components/__init__.py, nnterp/components/standard.py, nnterp/components/layer.py, nnterp/components/attention.py, nnterp/components/mlp.py, nnterp/families/gpt2.py, nnterp/families/llama.py, nnsight src/nnsight/intervention/envoy.py]
---

# Architecture

## What this is for

nnterp is three small layers on nnsight 0.8, and each owns one kind of fact.
A **family** (`nnterp/families/<model_type>.py`) knows the checkpoint's *names*
and *which operation* in its forward each value lives at. A **component**
(`nnterp/components/`) knows *what a value means* and how to read and write
it through nnsight. `StandardizedTransformer` (`nnterp/standardized.py`) knows
the *root*: the whole-model values, the methods over the values, the sizes
(their plain rule; a family supplies its own spelling), and the load path that wires the other two into nnsight's `rename=` and
`envoys=`. This page is the map; the other developing pages go one level
down into each box.

## Canonical pattern

The load path, observed on a tiny GPT-2 (run on
`hf-internal-testing/tiny-random-gpt2`; the ids in comments are what a user
types for the real model):

```python
import nnterp
from nnterp import StandardizedTransformer
from nnterp.families import gpt2

model = StandardizedTransformer("openai-community/gpt2", dispatch=True, attn_implementation="eager")

model.family is gpt2                       # True: config.model_type -> nnterp.families.gpt2
model._aliases                             # {'embed_tokens': 'transformer.wte', 'layers': 'transformer.h', 'norm': 'transformer.ln_f'}
model.layers[0]._aliases                   # {'input_layernorm': 'ln_1', 'self_attn': 'attn', 'post_attention_layernorm': 'ln_2'}
model.layers[0].self_attn is model.transformer.h[0].attn   # True: an alias is the same envoy
type(model.layers[0]) is gpt2.Layer        # True: the block is wrapped by the family's Layer
list(type(model.layers[0].self_attn).values())
# ['attention_queries_prerope', 'attention_keys_prerope', 'attention_queries', 'attention_keys',
#  'attention_values', 'attention_scores', 'attention_premix', 'attention_output',
#  'attention_probabilities', 'attention_head_outputs', 'interface']

with model.trace("The Eiffel Tower is in"):
    pattern = model.layers[2].self_attn.attention_probabilities.save()   # fires inside block 2 ...
    resid = model.layers[2].layer_output.save()                           # ... before the block returns
    logits = model.logits.save()
# resid [batch, seq, hidden_size]; pattern [batch, num_heads, seq, seq]; logits [batch, seq, vocab_size]
```

Every line after the load is a fact one of the three layers establishes; the
rest of this page says which.

## The map

```
 checkpoint / nn.Module
        │
        ▼  StandardizedTransformer.__init__                    nnterp/standardized.py
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ config = _read_config(repo_id, **kwargs)          AutoConfig, before any build │
 │ family = families.resolve(config)                   wrapper's type, else text's │
 │ rename = {**family.RENAME, **user rename}                                        │
 │ envoys = {**tp envoys (if sharded), **family.ENVOYS, **user envoys}              │
 │ TransformersModel.__init__(repo_id, rename=..., envoys=..., task=text-generation)│
 └──────────────────────────────────────────────────────────────────────────────┘
        │
        ▼  nnsight builds the envoy tree                        nnsight Envoy.__init__, envoy.py
   for each module: class = _resolve_envoy_class(module, path)   (type MRO, then path suffix)
   after the children exist: aliasing.bind(envoy)                (rename keys resolved per envoy)
        │
        ▼  the tree the user sees
   model                        StandardizedTransformer  ── logits, token_embeddings, next_token_probs,
   ├── embed_tokens ─┐ alias                                 input_ids, attention_mask, input_size,
   ├── layers ───────┤ alias of transformer.h                sizes, skip_layers/steer/project_on_vocab,
   │   └── [i]       │ family.Layer      ── layer_output      support()
   │       ├── self_attn   family.Attention ── attention_output, attention_probabilities, queries, ...
   │       ├── linear_attn family.LinearAttention (hybrids) ── attention_output, decays, betas, state*, ...
   │       └── mlp         family.Mlp        ── mlp_output
   │                       (a mixture: family.Moe or a Moe-based Mlp ── + router_logits, expert_weights/indices,
   │                        expert_outputs, routed_output, shared_expert_output)
   ├── norm ──────────┘ alias
   └── lm_head
        │
        ▼  inside a trace, a value read
   EProperty.__get__ ── _check(unavailable) ── _resolve(path) ── Mediator.value(location) ── _pick(select)
        key "output"                              → "{path}.output"                        (boundary values)
        key "../post_attention_layernorm.output"  → the sibling's ".output"                (a sibling's value)
        key "/inputs"                             → walked from envoy.root                 (a value far from the host)
        key "source.<call>.source.<op>.output"    → drilled per run → the op's location    (values inside a forward)
        │
        ▼  outside a trace, a size read
   StandardizedProperty.__get__ ── family.<name>(model) if the family module defines it, else the plain rule over config
```

## Two engines, one base

`Standardized` (`nnterp/standardized.py`) holds what does not depend on the
engine: the sizes, `support()`, `steer`, `skip_layers`,
`get_topk_closest_tokens`, `_read_config`. `StandardizedTransformer` mixes it
over `TransformersModel`, `StandardizedVLLM` (`nnterp/standardized_vllm.py`)
over nnsight's `VLLM`; each leaf resolves its family, merges `rename=` /
`envoys=`, and defines the root values where its engine keeps them.
vLLM's families are `nnterp/families/vllm/<model_type>.py`
(`lookup(model_type, engine="vllm")`), built from the bases in
`nnterp/components/vllm/` (`flat.py`, `layer.py`, `attention.py`, `mlp.py`,
mirroring `nnterp/components/`): `Flat`, an `EProperty` over a `[tokens, ...]`
tensor served as a private `[1, tokens, ...]` copy and handed back by a
transform (times a `factor`, divided back on a write, where the value is a
scaled copy), and `FusedLayer`, whose stream is the sum of the
`(hidden_states, residual)` pair a fused block takes and returns.
`transformers_backend.py` holds the bases of the families vLLM runs through
its transformers backend, where transformers' own modules run inside the
engine with the batch axis kept and the engine's attention layer outside
the module tree. The block
runs in the engine's worker against the client's envoy classes, pickled by
reference, the way a remote trace does. [usage/vllm](../usage/vllm.md).

## Data flow through `__init__`

`StandardizedTransformer.__init__` (`nnterp/standardized.py`) runs in
this order, and the order matters:

1. **Config first.** `_read_config` (`standardized.py`) returns a
   ready module's own `config`, or `AutoConfig.from_pretrained(repo_id, ...)`
   with the kwargs that say where the checkpoint is (`cache_dir`, `revision`,
   `token`, `local_files_only`, ...), so it is looked for where the build will
   look. It runs before nnsight builds anything, so
   a config transformers cannot parse fails here with transformers' own error,
   and so the family is known before the meta build.
2. **Family lookup.** `families.resolve(config)` (`nnterp/families/__init__.py`),
   which calls `lookup` on the config's own `model_type` when a family covers it (a
   wrapper with a family named after it, `kimi_k25`), else on its `text_config`'s: a
   multimodal config nests the language model's config there.
3. **Merge and hand to nnsight.** `rename={**self.family.RENAME, **(rename
   or {})}` and `envoys={**self._base_envoys(...), **self.family.ENVOYS,
   **(envoys or {})}` (`StandardizedTransformer.__init__`, `standardized.py`).
   Later keys win: a user's entry replaces the family's on the same key
   (`test_user_envoys_merge_over_defaults`, `tests/test_registry.py`).
   `_base_envoys` (`standardized.py`) starts from nnsight's tensor-parallel
   envoys when the load shards, because `TransformersModel` only *defaults*
   `envoys` to `tp_envoys()` (nnsight `TransformersModel.__init__`,
   `modeling/transformers.py`; `wants_tensor_parallel`,
   `modeling/tp/envoys.py`): passing any map of our own would otherwise drop
   them on a sharded load.
4. **`task` defaults to `"text-generation"`**, and `tokenizer_kwargs` become
   attributes on the loaded tokenizer (`StandardizedTransformer.__init__`,
   `standardized.py`).

## What nnsight does with the two maps

Both `rename=` and `envoys=` are nnsight features; nnterp only fills them in.
The two facts nnterp's design rests on:

- **The envoy class is chosen per module at construction.** `_wrap_envoy`
  calls `_resolve_envoy_class(module, child_path)` (nnsight
  `envoy.py`), which tries the module's `type(...).__mro__` against
  the map's type keys first, then string keys as dotted *path suffixes*
  (`path_ends_with`, `aliasing.py`), on the native path
  (`model.transformer.h.0.attn`) and then on each spelling of it under the
  `rename` aliases (`spellings`, `aliasing.py`). A type key always beats a
  path key. This is why every family keys `ENVOYS` on transformers' module
  types (`families/gpt2.py`) and why the README tells a user who wants to
  displace a family envoy to key on the type too.
- **Aliases bind after the children exist, per envoy.** `Envoy.__init__`
  wraps the children (`envoy.py`) and then calls `bind`
  (`aliasing.py`). Every envoy in the tree resolves every
  `rename` key relative to itself (`named`, `aliasing.py`); a key that resolves
  binds each alias as a plain attribute pointing at the *same* envoy object
  (recorded in `_aliases`), a key that does not resolve is skipped, and an
  alias that would displace a child, an `Envoy` attribute or a module
  attribute raises `ValueError`. The consequence nnterp relies on: a
  multi-component key (`"transformer.h"`) resolves only from the root, which
  is what lifts `layers` to `model.layers` (`RENAME`, `families/gpt2.py`),
  while a single-component key (`"attn"`) resolves in every block that has
  one. `test_no_inner_model_alias` (`tests/families/suite.py`) checks that
  nothing binds at `model.model`.

Because an alias is the same object, `model.layers[0].self_attn` *is* the
`gpt2.Attention` envoy nnsight built for `transformer.h.0.attn`, and the
descriptors on that class serve values at that module's native location.
nnsight's repr shows aliases as `alias/native` labels and lists every
eproperty that carries a `description` (`Envoy.__repr__`, `envoy.py`), which
is how `support()`-visible values show up in `print(model)`.

## Which layer owns what

| layer | owns | must not know |
|---|---|---|
| `nnterp/families/<model_type>.py` | `RENAME` (native name → standard name); `Layer`/`Attention`/`Mlp`/`Moe`/`LinearAttention` subclasses that point a value at *this family's* op or sibling; `ENVOYS` keyed on transformers types; a module-level `def <size>(model)` for each root size *this family's* config spells its own way (`falcon.py`: `num_kv_heads`, `intermediate_size`; `deepseek_v2.py`: `head_dim`, `qk_head_dim`) | what a value means, how nnsight serves it; the plain rule for a size |
| `nnterp/components/` | what each standard value **means** (`layer_output` is the residual stream leaving the block, `attention_output` the contribution, `attention_probabilities` the post-dropout pattern); how to read/write it (`EProperty` with a path for a key, `DerivedEProperty`); availability (`unavailable=`, `support`); the default op on transformers' shared interface (`INTERFACE`, `attention.py`) | any one family's module names or classes |
| `nnterp/standardized.py` | the root values (`logits`, `token_embeddings`, `next_token_probs`, `input_ids`, `attention_mask`, `input_size`); the methods (`skip_layers`, `steer`, `project_on_vocab`, `get_topk_closest_tokens`); the sizes (`num_layers` … `intermediate_size`, `standardized.py`), each a `StandardizedProperty` holding the plain rule over the config and yielding on read to a same-named function in `model.family`; `support()` over the tree (`Standardized.support`, through `standard_children` and `blocks_support` in `nnterp/components/standard.py`: each block's `Standard` children under their standard names, each alias read off its own binding on the block, a mounted alias such as DBRX's `norm_attn_norm.attn` and a module another block owns (shared weights) included, so a value installed through `envoys=` is listed and a module no block has is not); the remote key (`StandardizedTransformer._remoteable_class`) | op names inside a forward; any one family's config keys |

Two examples of the boundary. Gemma-2's contribution is the post-attention
norm's output: the *family* says so with an `EProperty` keyed
`"../post_attention_layernorm.output"` on its `Attention`
(`Attention.attention_output`, `families/gemma2.py`), while the *component*
only knows how to walk a path: `../` to the parent, a name to a child,
`source` into a forward (`EProperty._resolve`, `components/eproperty.py`).
BLOOM's contribution is the first argument of `dropout_add`: the family
names the op, `"source.dropout_add_0.input"` (`Attention.attention_output`,
`families/bloom.py`), the component drills to it and reads the call's first
argument. A family that keeps transformers' shared path overrides nothing
(`Layer`, `Attention`, `Mlp`, `families/llama.py`: three empty subclasses).
The sizes draw the
same line: the *root* holds the plain rule (`num_kv_heads` is
`config.num_key_value_heads` or `num_heads`, `standardized.py`), and
Falcon, whose config says `num_kv_heads`, `multi_query` or neither, says so
in a function of that name in its own module (`families/falcon.py`, the
`sizes` section at its end); `StandardizedProperty.__get__`
looks it up on `model.family` and calls it, or falls back to the rule;
`__set__` refuses an assignment, so the family module is the only
place a root size is said. A block's own sizes are plain properties of its
`Attention` (`num_heads`, `num_kv_heads`, `head_dim`, `qk_head_dim`) and `Mlp`
(`intermediate_size`), read off the module's attributes and projection shapes
rather than the config, so a family whose blocks differ needs no per-layer
config parsing; a module that spells one another way gets an override on the
family's subclass (`families/jetmoe.py`, `Mlp.intermediate_size`).

### The base envoys

`Standard` (`components/standard.py`) is the `Envoy` subclass every
component derives from: `values()` collects the `EProperty`s of a class,
base classes first, `support()` maps each to its reason on this instance,
and `sourced` (`False` here) is the flag a family sets to `True` on an
envoy whose forward holds a value read after the call has started;
`__init__` and `_update` then instrument that forward at build and again
when real weights replace meta ones (Llama 4's `Layer`, for its
`Mlp.mlp_output`). `Layer` (`components/layer.py`) adds `returns_tuple`,
`skip_with` and `layer_output` (`first_tensor` in, `rewrap` out).
`Attention` (`components/attention.py`) adds the contribution and the six
interior values on `INTERFACE`; `off_interface` is the one method a family
overrides to give another reason the interface does not run
(`families/gpt2.py` for `reorder_and_upcast_attn`). `Mlp`
(`components/mlp.py`) adds `mlp_output`. `Moe` (`components/moe.py`), an
`Mlp`, adds a mixture of experts' six values, read where the model consumes them:
`router_logits` at the router's logits op (`LOGITS`, `F_linear_0`), the routing pair
as the experts module's arguments, `expert_outputs` inside transformers'
`grouped_mm` / `batched_mm` experts forward (`DISPATCH`, `PER_SLOT`),
`routed_output` as the experts' output, `shared_expert_output` as the shared
expert's. The model routes flat `[batch * seq, ...]` tensors; the values are
`TokenEProperty`s (`components/tokens.py`), an `EProperty` subclass that serves
this invoke's `[batch, seq, ...]` rows of what `EProperty` reads (`rows`, from the
interleaver batcher's `total` and the worker's `batch_group`) and splices a write
back into the flat tensor before `EProperty` writes it (`splice`).
`no_mixture` (`None` on the base) is the one method a family overrides when its
host runs no mixture on some checkpoints (Gemma-4's dense MLP, which hosts the
block's experts; GraniteMoE-Hybrid's shared MLP; Doge), as `off_interface` is for
attention; `num_experts`, `top_k` and `SCORING` are its sizes and its router's kind.
`RecurrentMixer`
(`components/recurrent.py`) adds `attention_output`, the kernel choice, the
per-token state and the kernel routing for a mixer read at a kernel call, and
`LinearAttention` (`components/linear_attention.py`) sets its kernel
constants and declares the gated DeltaNet's values on it; both are their own
page, [recurrent-mixer-internals.md](recurrent-mixer-internals.md).

## The registry

`nnterp/families/__init__.py` is the whole registry, and it is lazy on
purpose: `import nnterp` must import no transformers modeling module
(`test_import_is_lazy` in `tests/test_registry.py` runs that in a subprocess).

- `known()` (`families/__init__.py`) lists the package's modules with
  `pkgutil.iter_modules`, but `default`; that list is the set of shipped
  families, and a module's name *is* its `model_type`
  (`test_every_family_module_is_named_after_its_model_type`,
  `test_registry.py`).
- `lookup(model_type)` returns `REGISTRY[model_type]` when
  something was `register`ed, else `importlib.import_module(f"nnterp.families.{model_type}")`.
  A `ModuleNotFoundError` whose `.name` is that exact module means "no such
  family": `lookup` warns and returns `nnterp.families.default`; any other
  `ModuleNotFoundError` is a family module that itself failed to import, and
  is re-raised as the real error.
- `default` is a family like the others that `lookup` returns, with a warning, when
  neither of the above has the `model_type`. Its
  `RENAME` is the union of the shipped spellings, its `ENVOYS` are string keys
  on the standard names (nnsight matches them on alias paths; `"layers.*"`
  reaches the blocks), and it defines `check(model)`, which
  `StandardizedTransformer.__init__` calls on the built tree when the family
  is `default` (by identity: a copy skips it): the required root modules and
  a shape-only `scan` of the stream and the logits, raising
  `UnsupportedFamily` when the guess cannot stand or the scan cannot run. It
  serves `layer_output`, the root values, the sizes and the interface
  interior; `attention_output`, `mlp_output` and `project_on_vocab` are
  `Unavailable` with one reason each, since what a sublayer adds and what
  follows the head are code a name does not show.
- `register(family, *model_types)` writes each of `model_types`, or with
  none the last component of `family.__name__`, into `REGISTRY`, which
  `lookup` consults first, so a module from outside the package, or an
  override of a shipped one, needs no edit here
  (`test_register_adds_a_family_and_can_override`,
  `test_register_takes_the_type_from_a_modules_name`, `test_registry.py`).
- `__getattr__` makes `nnterp.families.qwen3_5_text` import on first
  attribute access, and `__dir__` lists the shipped names so tab completion
  works before anything is imported. `all_families()` imports everything,
  for tooling and tests only.

A family module imports its transformers modeling module at the top
(`families/gpt2.py`), so the modeling module loads exactly when the first
checkpoint of that type is looked up, and never earlier.

## Where it lives

| concern | file |
|---|---|
| load path, root values, methods, sizes and `StandardizedProperty`, `support()` | `nnterp/standardized.py` |
| registry, `lookup`, `register`, `UnsupportedFamily` | `nnterp/families/__init__.py` |
| one family | `nnterp/families/<model_type>.py` |
| descriptors | `nnterp/components/eproperty.py` ([eproperty-internals.md](eproperty-internals.md)) |
| base envoys | `nnterp/components/{standard,layer,attention,mlp}.py` |
| recurrent mixers: the base, and the DeltaNet subclass | `nnterp/components/recurrent.py`, `nnterp/components/linear_attention.py` ([recurrent-mixer-internals.md](recurrent-mixer-internals.md)) |
| forward order (`order()`, `rank()`), measured by one probe run per model and block shape | `nnterp/order.py` |
| helpers that use the values | `nnterp/prompt_utils.py`, `nnterp/nnsight_utils.py` |
| the executable contract | `tests/families/suite.py` ([testing.md](testing.md)) |

## Gotchas

- `envoys=` matches type before path (a path key matches the native path or
  any alias spelling of it); a family's type key can only be displaced by a
  type key (nnsight `Envoy._resolve_envoy_class`, `envoy.py`).
- The family is chosen from `config.model_type` *before* the model is built,
  so an already-loaded `nn.Module` is looked up by its own `config`
  (`Standardized._read_config`, `standardized.py`;
  `test_preloaded_module_uses_its_own_config`, `tests/test_registry.py`).
- Passing `envoys=` of your own to a plain `TransformersModel` drops
  nnsight's tensor-parallel envoys; `StandardizedTransformer` starts from
  them (`StandardizedTransformer._base_envoys`, `standardized.py`), so do
  the same in a subclass.
- Reads inside one trace follow forward order, and a value inside a block
  (`attention_probabilities`) fires before the block's `layer_output`; the
  canonical example reads them in that order. Read out of order and the
  trace raises `OutOfOrderError` naming the attention call's `.fn`: see
  [gotchas.md](gotchas.md).
- A size override is a function on the family *module*, looked up by name on
  `model.family` at read time; a family object passed to `register()` carries
  it as an attribute (`test_family_defines_a_size_instead_of_the_root`,
  `tests/test_registry.py`), and a variant that spreads a shipped family's
  dicts leaves the shipped functions behind.

## Related

- [eproperty-internals.md](eproperty-internals.md) — the four descriptors and the nnsight facts they use
- [recurrent-mixer-internals.md](recurrent-mixer-internals.md) — `RecurrentMixer`, the DeltaNet subclass, and occurrence arithmetic
- [testing.md](testing.md) — `FamilySuite`, the contract every family passes
- [transformers-compat.md](transformers-compat.md) — what a release can rename
- [gotchas.md](gotchas.md) — contributor-facing traps
- nnsight `docs/usage/rename-modules.md` and `docs/usage/source.md` — the two features nnterp builds on
- nnsight `docs/developing/extending-envoy.md` — the `eproperty` surface nnterp's descriptors subclass
