# Naming and reaching tensors: a design for nnterp's internals layer

## What shipped

This document is the proposal as it was written. What the branch actually shipped is
smaller, and where the two disagree the code is right:

* **`Address` + `LayerAccessor` + `model.internals`** shipped, as did availability
  answered before any trace (`internals.status()`, per layer where a model's layers
  differ), forward order (`Address.order`, `internals.rank`), `model.block_structure`,
  the six block-interior places, the whole-model places as rows, and the published
  sizes. `RenameConfig(addresses=...)` is the way in from outside.
* **`Selection` instead of `Lens`**: where a tensor sits inside a value is a
  `Selection` — `Path(*steps)` or `FirstIfTuple()`, or one of your own — on
  `Address.select`, not the get/put pair this document sketched.
* **Not shipped**: `Layout`/`Axis`/`Fused` and `internals.layout(...)`, per-head
  indexing (`[i, h]`), `attention_z` and the other attention interiors, the ordering
  guard and `internals.read(3, *names)` (`internals.rank` sorts, and the caller reads),
  `match_op` (operations are named literally), and the version-keyed table.
* `design/prototype/` is the proposal's standalone prototype, not nnterp's code: its
  `Address` has different fields from the one that shipped.

Status: proposal. Branch `standardize-internals`. Nothing in `nnterp/` is changed by this
document; the prototype under `design/prototype/` is standalone and runnable, and
`design/prototype/demo_output.txt` is its recorded output on `Maykeye/TinyLLama-v0`, `gpt2`
and `hf-internal-testing/tiny-random-MistralForCausalLM` (CPU, transformers 5.15.0.dev0,
nnsight 0.8.0).

nnterp's job today is naming modules. The proposal is to grow it to **naming and reaching
tensors**: "the query vector of layer 3 is here, and here is how to get it, on any supported
architecture." Everything about meaning and policy — a closed component vocabulary, a
normalized tensor layout, write legality, intervention mechanics, metrics — stays downstream.

---

## 1. The accessor API

Two surfaces. The indexed accessors keep the spelling nnterp's users already know:

```python
model = StandardizedTransformer("meta-llama/Llama-3.1-8B", enable_attention_probs=True)

with model.trace(prompt):
    q   = model.attention_queries[3]            # read
    p   = model.attention_probabilities[3]
    model.attention_z[3, 7] = replacement       # write head 7's output vector
```

and a registry namespace carries everything *about* them:

```python
model.internals.names                 # what this model supports
model.internals.status()              # a table: name, available?, layout / reason
model.internals.layout("attention_z") # the descriptor (section 4)
model.internals.read(3, *names)       # read several, in forward order, whatever order you ask
model.internals["attention_gate"]     # the accessor, or a RenamingError with the reason
```

The full set, with `[i]` the layer and `[i, h]` the head:

| accessor | what it is |
|---|---|
| `attention_queries_pre_rope`, `attention_keys_pre_rope`, `attention_values` | the projections, before RoPE. On a fused tree these are column views of one native tensor. |
| `attention_gate` | the `[q \| gate]` split, gated-attention families only |
| `attention_queries`, `attention_keys` | post-RoPE; `keys` is pre-`repeat_kv`, so KV-head space |
| `attention_scores` | the post-mask softmax **input** |
| `attention_probabilities` | the softmax **output** (existing name, existing semantics) |
| `attention_z` | per-head attention output, `(b, s, H, d)`, pre `o_proj` |
| `attention_premix` | the same tensor head-major-flat: `o_proj` / `c_proj` input |
| `mlp_act_fn_output`, `mlp_neuron_outputs` | see section 9.1 — two names, never collapsed |
| `router_logits`, `router_scores`, `router_indices` | the MoE router's three returns |
| `expert_gate_proj`, `expert_up_proj`, `expert_activations`, `expert_neuron_outputs`, `expert_outputs` | the per-expert interior |
| `deltanet_*` | the linear-attention mixer's interior |

**Per-head spelling: an index, not a separate accessor, not a slice the user computes.**
`model.attention_z[3, 7]` and `model.attention_keys[3, 1]`. The bound comes from the
*component's own* head space, read off its layout, never from `model.num_heads`. Under GQA a
query-space index over a KV-space component produces an **empty** slice and Python does not
raise; the prototype refuses instead (measured on tiny Mistral, H=4, KV=2):

```
[ok] GQA: attention_keys[0, 3] refused: attention_keys names head 3, but this component
     has 2 heads (kv_head space, layout (batch, kv_head, position, head_feature))
```

A separate `attention_z_heads` accessor was rejected: it doubles the name count, and the head
axis is not a different tensor, it is an index into the same one. A user-computed slice was
rejected because computing it correctly requires knowing the packing, which is exactly what
nnterp is for.

Writes go through the same subscript. `accessor[i] = tensor` replaces the whole value;
`accessor[i, h] = tensor` and `accessor[i][..., h, :] = 0` both edit in place. nnterp already
carries `num_heads`; it must gain `num_kv_heads` and `head_dim`, which it has **nowhere** today
(no occurrence of `num_key_value_heads` or `head_dim` anywhere in the repo).

---

## 2. The address table

One row per (family tree, name). Pure data.

```python
@dataclasses.dataclass(frozen=True)
class Address:
    module: str                       # dotted path under the layer envoy; "" is the layer
    op: tuple[str, ...] = ()          # `.source` patterns, one drill between elements
    handle: Literal["output", "input", "inputs"] = "output"
    select: tuple[int | str, ...] = ()  # indices into the handle; (0, 1) on inputs is arg 1
    layout: Layout = ...              # section 4
    order: int = 0                    # rank in the block's forward; section 5
    requires: frozenset[str] = ...    # capability flags; section 3
    writable: bool = True
    attention_row: bool = False       # softmax-attention blocks of a hybrid only
```

`module` is walked on the **envoy** tree, so it speaks nnterp's standardized names
(`self_attn`, not gpt2's `attn`). Populated tables for both trees are in
`design/prototype/internals.py`; the shape is:

| name | `llama_tree` | `gpt2_tree` | order |
|---|---|---|---|
| `attention_queries_pre_rope` | `self_attn.q_proj`.output | `self_attn.c_attn`.output, block 0 of 3 | 10 |
| `attention_keys_pre_rope` | `self_attn.k_proj`.output | `self_attn.c_attn`.output, block 1 of 3 | 11 |
| `attention_values` | `self_attn.v_proj`.output | `self_attn.c_attn`.output, block 2 of 3 | 12 |
| `attention_queries` | `self_attn` → `attention_interface` → `inputs[0][1]` | *same row* | 20 |
| `attention_keys` | `self_attn` → `attention_interface` → `inputs[0][2]` | *same row* | 21 |
| `attention_scores` | `self_attn` → `attention_interface` → `nn_functional_softmax` → `inputs[0][0]` | *same row* | 30 |
| `attention_probabilities` | `self_attn` → `attention_interface` → `nn_functional_softmax` → `output` | *same row* | 31 |
| `attention_z` | `self_attn` → `attention_interface` → `output[0]` | *same row* | 40 |
| `attention_premix` | `self_attn.o_proj`.input | `self_attn.c_proj`.input | 41 |
| `mlp_act_fn_output` | `mlp.act_fn`.output | `mlp.act`.output | 60 |
| `mlp_neuron_outputs` | `mlp.down_proj`.input | `mlp.c_proj`.input | 61 |

Five of eleven rows are literally shared between the two trees, because both mixers call the
same `ALL_ATTENTION_FUNCTIONS` interface. That sharing is the payoff of anchoring on the call
rather than on the family.

The `linear_attn` rows (`deltanet_conv`, `deltanet_queries/keys/values`, `deltanet_beta`,
`deltanet_decay`, `deltanet_kernel_output`) are in the prototype table in the same format,
anchored on `linear_attn` instead of `self_attn`. They are **design rows, not verified here**:
the only hybrid checkpoint in this environment's cache has a config but no weights, and
nnterp refuses an all-linear model anyway (`_check_attention_layers` requires at least one
`self_attn`). `deltanet_state` is deliberately *not* an `Address` — it fires once per 64-token
chunk, which the one-value-per-`[i]` contract cannot express; it gets its own accessor type
returning a sequence, exactly as the inventory recommends.

### `match_op`

Taken from causalab's `nnterp_engine/sources.py`. Substring match over the installed forward's
op names; on several hits, prefer the ones whose own source line *calls* the matched symbol; on
zero or still-several, refuse with the full op inventory printed. It resolves the
assigned-then-called ambiguity structurally — which is exactly what commit `41f8573` on this
branch repaired by hand (`attention_interface_0` → `_1`, twice).

Two honest qualifications:

* **`match_op` does not resolve every multi-hit case, and should not.** GPT-2's mixer carries
  `self_c_attn_0` (the dead cross-attention branch) and `self_c_attn_1` (the live one); both
  lines read `self.c_attn(`, and the call-preference test looks for `self_c_attn(`, which
  appears in neither. `match_op` would refuse with two hits. That is correct behaviour — a
  branch-dead op is not something a matcher can rank — and the table sidesteps it by addressing
  `c_attn` at the module boundary, where there is one envoy and one `.output`.
* **Licence.** `sources.py` imports nothing from causalab (enforced by a subprocess test there),
  so the code move is a file copy. The *legal* move is not free: causalab ships no `LICENSE`
  file and declares no licence in `pyproject.toml`, so it is all-rights-reserved by default,
  while nnterp is MIT (© 2024 Clément Dumas). Landing `match_op` and the table needs an explicit
  grant from causalab's authors (its `pyproject.toml` names Atticus Geiger and Max Loeffler) to
  relicense under MIT, plus a provenance line in the file header. One email, but do it first.

### Address by the call, not by the variable binding

causalab's table spells the softmax neighbourhood as `attn_weights_1` (scores) and
`attn_weights_2` (probs). Measured on transformers 5.15, those suffixes do **not** mean the same
thing on the two trees:

* llama's `eager_attention_forward`: `attn_weights_2` is post-softmax **and** post-`.to(dtype)`,
  `attn_weights_3` is post-dropout;
* gpt2's: `attn_weights_2` is post-softmax **pre**-cast, `_3` is post-cast, `_4` is post-dropout,
  because gpt2's copy has an extra `attn_weights.type(value.dtype)` line.

Both happen to be valid probabilities, so the table works — by luck. The suffix on a *variable
binding* counts assignments in the whole function, so any added line shifts it. The suffix on a
*call op* (`nn_functional_softmax_0`, `nn_functional_dropout_0`) counts only calls of that
symbol, of which there is normally exactly one. So the table addresses the softmax's `output`
for probabilities and its `inputs[0][0]` for scores, and the two remaining hardcoded suffixes
disappear. This is a change from what both prior reports propose, and it is the single cheapest
drift reduction available.

---

## 3. Availability, decided before the first trace

One uniform story, generalizing `attn_probs_available` / `RenamingError` / `.disable()`.

Two kinds of question, both answered at load with no trace:

1. **Is the module there?** Walk `address.module` on the layer envoy. A miss yields the path
   that failed and the children that exist.
2. **Is the capability there?** `attn_eager` (config `_attn_implementation`), `gated_attention`
   (q-projection `out_features == 2·H·d`), `moe` (`mlp` has `gate` and `experts`),
   `experts_grouped` (`config._experts_implementation`). These are causalab's `_probe_*`
   predicates, which are architecture questions with architecture answers.

The accessor is then constructed or not. `model.internals.status()` is what a user asks:

```
tree: gpt2_tree    capabilities: attn_eager=yes, gated_attention=no, moe=no
  [ok]   attention_queries_pre_rope   (batch, position, feature[head][blocks 0/3])
  [ok]   attention_probabilities      (batch, head, position, key_position)  rows sum to 1
  ...
```

and touching what is not there raises a `RenamingError` carrying the reason, never an
`AttributeError`:

```
attention_gate is not available on this model: this mixer computes no output gate: the box
exists only on the gated-attention family, whose q-projection emits [q | gate] per head.
```

Why this and not `eproperty`: an `eproperty` can only fail *at access time, inside a trace*, and
an `AttributeError` raised in its preprocess is swallowed by `property` and rewritten into "no
such attribute". Answering before the trace is nnterp's whole value proposition.

A separate policy question, which must be settled before twenty names multiply it: **an
architecture that legitimately lacks a component is not a test failure.** Today
`test_probabilities.py` calls `pytest.fail` rather than skip when probs are unavailable, so the
status file records it. With twenty components, "this family has no MoE" would paint the matrix
red. The rule: *missing capability* → recorded in the `no_X_available_models` bucket, not a
failure; *capability present but address unresolvable* → failure.

---

## 4. Layout: native, plus a descriptor

nnterp returns the tensor as transformers produced it and ships a descriptor beside it. It never
permutes, never reshapes, never flattens a head axis, never introduces one.

```python
@dataclasses.dataclass(frozen=True)
class Axis:
    kind: Literal["batch", "position", "key_position", "head", "kv_head",
                  "head_feature", "feature", "expert", "top_k", ...]
    heads: Literal["head", "kv_head"] | None = None   # a feature axis that flattens heads
    fused: Fused | None = None                        # ... and packs several logical tensors

@dataclasses.dataclass(frozen=True)
class Fused:
    splits: int
    index: int
    mode: Literal["blocks", "per_head"]   # contiguous blocks, or interleaved per head

@dataclasses.dataclass(frozen=True)
class Layout:
    axes: tuple[Axis, ...]
    note: str = ""
```

That is the whole descriptor, and it says four things: **what each axis is**; **which head space
a flattened feature axis belongs to** (the GQA trap, since a flat `H·d` and a flat `H_kv·d` look
identical); **which logical tensor of a fused projection this is, and how the fusion is laid
out**; and — where a kernel permutes rows — **which address gives the un-permutation**
(the MoE `align` field; not exercised in the prototype).

A downstream library normalizes from `axes` alone and never asks nnterp what normalization
means: `layout.index_of("head")` gives the permutation source, `heads` tells it a feature axis
can be viewed apart, `fused` tells it a write shares a native tensor with siblings. causalab's
`layout.py` becomes a *consumer* of this and stays in causalab.

**One deliberate divergence from "returns native".** On GPT-2, `attention_queries_pre_rope`
returns the 768-wide **view** of `c_attn`'s 2304-wide output, not the 2304-wide tensor. The line
I draw: nnterp may *select* (a view — no copy, no permutation, no reshape, writes land in the
right columns and the siblings survive); it may not *permute or reshape*. Returning the whole
fused tensor under a name that means "the queries" would be a worse answer than a view, and the
layout still records the fusion so nothing is hidden. Measured:

```
[ok] attention_queries_pre_rope[0] = 0 moved the logits: True;
     attention_values unchanged by it: True (shares a native tensor: True)
```

---

## 5. Ordering

nnsight refuses a request for a value the forward has already passed. Every row carries an
`order` int — its rank inside the block's forward — and reads must be non-decreasing in
`(layer, order)`.

Three things follow, and the API does all three:

1. **`model.internals.read(layer, *names)` sorts.** This is the spelling a user reading five
   internals should reach for, and it is why they never need to know the execution order. The
   prototype reads five, named in a deliberately wrong order, and returns all five.
2. **A manual out-of-order access refuses immediately, naming the right order.** Not at teardown
   (where nnsight's own `OutOfOrderError` surfaces, naming a location like
   `model.model.layers.0.self_attn.output.i0` that the user never wrote), but at the line:

   ```
   attention_probabilities[0] comes before attention_z[0] in the forward, which this trace
   has already read: nnsight cannot reach back to a value the model has passed.
     this trace asked for: ['attention_z', 'attention_probabilities']
     forward order is:     attention_probabilities then attention_z
     either read them in that order, or let nnterp sort for you:
     model.internals.read(0, *['attention_z', 'attention_probabilities'])
   ```
3. **It does not silently sort the manual path.** Reordering a user's statements would change
   intervention semantics — a write to the queries followed by a read of the probabilities is
   not the same experiment as the reverse — so the manual path refuses and the batch path
   sorts. Both exist on purpose.

The guard covers **writes** as well as reads, which the inventory frames as a read-side concern.
It caught two genuine out-of-order sequences while the prototype was being written, one of them
a write.

Implementation note: per-trace state is keyed on the nnsight `Mediator` object held by strong
reference. Keying on `id()` is wrong — ids are reused once the previous trace is collected, so a
fresh trace looks like a continuation of the last one, and the guard fires spuriously. That cost
one debugging round.

---

## 6. Writes

Two paths, chosen by whether the write is partial:

* **Whole value** → swap through the handle, rebuilding whatever container `select` indexed
  (a tuple return, an argument list, a kwarg). This is what `LayerAccessor.__setitem__` already
  does for tuple outputs.
* **Partial** (a head, a fused split) → in place on the view, which lands with no write-back
  machinery because the view aliases the model's own tensor.

**The eproperty report's "keep reshapes as views" is necessary but not sufficient.** Measured:
`attention_z[0, 1] = 0` in place works on Llama, and torch **refuses it on GPT-2** —
`RuntimeError: Output 0 of Transpose is a view and is being modified inplace. This view is the
output of a function that returns multiple views.` Llama's copy of the function ends with
`.contiguous()`; GPT-2's does not. So the accessor tries in place and falls back to
clone-edit-swap. Without that fallback, half the per-head writes in the design are dead on
arrival on GPT-2, and the failure is a torch error a user cannot act on.

### The check that must run, and what it costs

The trap that matters is a plausible address that reads perfectly and is causally inert. The
mixer *does* return the attention weights, so `self_attn.output[1]` /
`attention_interface.output[1]` has the right shape, the right values, and rows that sum to 1 —
and writing it changes nothing, because the value-weighted output is already computed. The
prototype carries that address as a named negative control and the check catches it on both
trees:

```
[!! ] probs_at_mixer_return   shape (1, 16, 11, 11)  declared (1, 16, 11, 11)  causal=False
inert addresses caught: ['probs_at_mixer_return']
```

Two tiers, because they cost differently:

* **Structural, at load.** One trace (or scan) for *all* rows, in forward order: each address
  resolves, and its native shape matches its declared layout. Measured: **0.02 s for 12 rows**
  on a tiny model. This is affordable at load and generalizes `check_io`.
* **Causal, in CI.** One trace *per row* — you cannot attribute a logit change to a write you
  did not isolate. Measured: **0.10 s for 12 rows** on a tiny model, i.e. O(rows) forward passes.
  On a real model that is not a load-time cost, and it is exactly the thing that has to be
  recorded per (transformers, nnsight, architecture) rather than recomputed. See section 7.

That split is the honest answer to the inventory's §3g: `check_source`'s two-trace causal
assertion, which nnterp runs at load today for one value, does not scale to twenty, and becomes
a test-matrix artifact consulted at load.

---

## 7. Drift, and why this time is different

The history has to be stated plainly, because it argues against this proposal:

* nnterp built a version-keyed capability table and **deleted the consumer** (`85a8c6a`:
  `warn_about_status`, `CLASS_STATUS`, `_get_closest_version`, 167 lines out of `utils.py`),
  and dropped `data/` from package-data in the same commit.
* `docs/model-validation.rst:119-124` still describes the deleted feature.
* `MANIFEST.in` ends with `recursive-exclude nnterp/data *.json`, `.gitignore` ignores
  `nnterp/data/test_logs/`, and nothing ships.
* The deleted consumer was called as `warn_about_status(model_name, self._model, model_name)`
  where `model_name` is the repo id for the ordinary `StandardizedTransformer("gpt2")` path,
  while `CLASS_STATUS` is keyed on architecture class names (`GPT2LMHeadModel`). Every lookup
  missed. **`85a8c6a` deleted something that largely was not working** — a weaker precedent
  against re-introducing it than the commit message implies.
* The producer survived intact: `conftest.py` still writes
  `{transformers}/{nnsight}/{bucket}/{ArchClass}: [repo_ids]`, and the buckets are *generated*
  from `test_config.yaml`'s `test_file_categories`, so a new capability adds
  `failed_X_models` + `no_X_available_models` for free.

What is different:

1. **The addresses stop being literals.** Today they are six hardcoded op names dispatched by an
   `isinstance` ladder, and the last five commits on this branch are each a hand-repair of one
   of them — including one (`47b41ae`) that *removed* the multi-candidate fallback `84c1cbc` had
   added. Patterns make the same drift loud and self-describing: `AddressResolutionError` naming
   the pattern, the hit count, and the installed forward's entire op inventory. A wrong answer
   becomes an error instead of a neighbouring tensor.
2. **The table becomes the only copy.** One data structure, not six functions.
3. **The key bug is fixed and the key is widened.** Key on
   `(transformers version, nnsight version, architecture class)` — `tests/utils.py` already
   computes the arch class, which is what the table has always been keyed by; pass *that*, not
   the repo id. Buckets widen from one attn-probs flag to one per component.
4. **What ships.** `nnterp/data/test_logs/latest_status.json` goes back into
   `[tool.setuptools.package-data]` and out of `MANIFEST.in`'s exclude. Current size: 16 KB.
5. **What a user sees when transformers moves ahead**, in three tiers:
   * exact `(transformers, nnsight, arch)` hit → each accessor enabled or `disable()`d with a
     reason naming the component and the version;
   * no exact hit → accessors stay **enabled**, one info line naming the nearest tested version.
     Never a hard failure for an untested version; nnterp's whole posture is "if it loads it
     probably works", and the pattern matcher is the safety net;
   * a pattern that fails to resolve at access → `AddressResolutionError` with the inventory.
6. **Precondition, stated honestly.** `.github/workflows/` contains `claude.yml` and `docs.yml`
   and **nothing runs the test suite**. A version-keyed table whose producer is "a developer
   remembers to run pytest locally" is the previous failure repeated with more rows. Step 0 of
   the migration is a workflow that runs the matrix across a transformers grid. Without it,
   ship steps 1–3 and skip step 4.

Tested by: a canary test that resolves **every** row for the detected family in rank order, one
trace per anchor, failing with the op-inventory diff; plus the causal check per row per model in
the matrix, whose results are what the status buckets record.

One more blind spot to fix while moving: tiny fixtures can be degenerate. A fixture where all
three MLP inner widths coincide cannot distinguish a right config-key choice from a wrong one —
and `mlp_act_fn_output == mlp_neuron_outputs` is *true on GPT-2 by construction*, so a test that
asserts they differ would be wrong, and one that asserts they agree would pass vacuously on the
wrong address. The table should carry the widths it expects and assert them distinct only where
the real config makes them distinct.

---

## 8. Migration, ordered by value for effort

| step | lands in nnterp | causalab deletes |
|---|---|---|
| **0** | a CI workflow that runs the suite on a transformers grid | — |
| **1** | `match_op` + the six existing attention-prob addresses re-expressed as patterns. **No new names.** Makes the feature nnterp already ships drift-loud instead of drift-silent. | — |
| **2** | the five shared attention-interface rows (`queries`, `keys`, `scores`, `probabilities`, `z`), the `Address`/`Layout` dataclasses, availability, ordering | `sources._ATTENTION`, `_attention_interior_site` |
| **3** | module-boundary rows, `num_kv_heads`/`head_dim`, per-head indexing, fused splits | `_measured_address`, `_interior_address`, `_projection_width`, `_has_separate_projections`, `_probe_split_qkv`, `_probe_gated_attention`; the `overrides`/`Packing`/`_check_override` half of the registry; `TreeAddress`/`LLAMA_TREE`/`GPT2_TREE`; most of `model_info_from_hf_config` |
| **4** | the version-keyed capability table (needs 0) | — |
| **5** | MoE grouped-kernel interior: `align`, `expert_rows`, `derive`, the per-trace op memo | `sources._MOE`, `_probe_moe`, `_probe_grouped_mm`, `Navigation`/`_op`/`_handle`/`_perm` |
| **6** | DeltaNet rows + the per-fire accessor type | `sources._DELTANET`, `GENERATED_ADDRESSES`, `fire_ops` |

Step 1 is the highest value per line by a wide margin, because it fixes something nnterp already
ships and costs causalab nothing.

**What stays in causalab forever.** The 56-name closed vocabulary and `Capability`'s
`writes`/`why`/`reason`/`expert_selection` cells. `layout.py` entire, as a consumer of the
descriptor. `COMPONENT_RANK` *as used for group elision*. The featurizer legality rules and the
"top-k is a ranking, not a basis" refusal. The write math: `block_mid`'s `writeback` as a delta
on `block_output`, `attention_result`'s derivation from the o-projection's weights, the
`additive` rule. The residual-identity **tolerances**. Mechanisms, metrics, `BACKEND_PAIRS`,
`DELTA_KERNEL_SLOTS`, and the `ProtocolError` reason codes.

---

## 9. What nnterp should not take

Not the closed vocabulary — a fixed list of 56 names with aliases is a contract between causalab
and its documents, and nnterp's list should grow when a family needs a name. Not the
`(batch, position, feature)` contract layout — it is causalab's math contract, it is not total
(the attention pattern has no contract form and the DeltaNet state bypasses the flattening), and
a library that normalizes away the head axis its consumers want does negative work. Not write
policies — "a write to `router_logits` reaches nothing" is a dataflow fact nnterp can carry as
`writable=False`, but "therefore refuse, and here is the alternative component" is policy. Not
execution ranks as a global 56-name table — nnterp publishes a block-local `order` per row; the
global rank exists to drive causalab's group elision. Not mechanisms, not metrics, not
`expert_selection`, not the reason-code taxonomy.

### The six ambiguous rows

**1. `mlp_activation` — and I disagree with the inventory's reading of it.** The inventory says
llama's `act_fn`.output and gpt2's `c_proj`.input are "genuinely different tensors" (`act(gate)`
vs `act(gate)·up`). Measured: GPT-2 has no gate, so `c_proj`.input **is** `act`.output — the two
addresses name the same tensor there. The divergence is not between the families under one name;
it is between **two names** that coincide on an ungated MLP and differ on a gated one:

```
Maykeye/TinyLLama-v0: act_fn output (1,11,256) vs neuron output (1,11,256): identical = False
gpt2:                 act output   (1,10,3072) vs neuron output (1,10,3072): identical = True
```

So the resolution stands, for a better reason: nnterp ships **both** faces under unambiguous
names — `mlp_act_fn_output` (the activation function's output) and `mlp_neuron_outputs` (the
down-projection's input) — declares nothing about their relationship, and lets causalab alias
`mlp_activation` to whichever its history requires. nnterp never asserts they are the same and
never asserts they differ; on GPT-2 both are true of different pairs.

**2. `expert_rows`.** Split as the inventory proposes. Un-sorting the kernel's permutation is
reaching — nobody wants rows in a kernel-internal order — so nnterp owns `align`. Re-packing
`(b·p·top_k, …)` into `(b·p, top_k·…)` is already the contract's shape, so causalab owns it.

**3. `COMPONENT_RANK`.** The inventory calls publishing-vs-documenting a genuine fork. I take
the publishing side: a per-row block-local `order` int. Documenting "request in forward order"
puts the model's internal execution order in the user's head, which is the thing this layer
exists to remove; and the sorted read and the named-order error both need the numbers, not prose.
causalab keeps its global 56-name rank for elision — a different artifact for a different job.

**4. Residual identities and tolerances.** Split. nnterp owns the *identities* as validation —
it already does a weaker value-based version — but expressed as the check it can actually
afford: "does a write here move the logits", per row, in CI. The fp32-exactness tolerance table
is a measurement causalab made for its own tests and stays there.

**5. `attention_scores` as the post-mask softmax input.** Within nnterp's remit — it already
pins `attention_probabilities` to one op and validates the pin causally. Better: address it as
the softmax call's `inputs[0][0]`, which makes the pin structural rather than a choice between
two suffixes of the same variable name.

**6. `DELTA_KERNEL_SLOTS`.** Stays in causalab. The kernel *boundary* is architecture and nnterp
keeps its `.source` twin; the ten slot names exist because causalab's reference engine swaps
modeling-file globals, a mechanism nnterp does not have and should not acquire.

---

## Appendix: what the prototype proves, and what it does not

Proved, on `Maykeye/TinyLLama-v0`, `gpt2` and tiny Mistral (full log in `demo_output.txt`):
one table over two different trees; `match_op` resolving assigned-then-called and refusing a
moved forward with the inventory; availability before the first trace, including the GPT-2 rows
resolving only once the walk is on the envoy tree; five internals read in one trace named in the
wrong order, sorted by `read()` and refused with the right order by the manual path; a per-head
write and a fused-split write that move the logits while leaving the siblings alone; the GQA
head bound refusing a query-space index over a KV-space component; the inert address caught at
check time on both trees; the structural/causal cost split; `remote="local"` reading and writing
through the accessors; and the accessor classes, table and layouts pickling **by value**
(53 KB), which is the property `eproperty` fails.

Not proved: the `linear_attn` rows (no weights available offline, and nnterp will not load an
all-linear model); the MoE grouped-kernel rows and the `align` un-sorting; the per-fire accessor
type for `deltanet_state`; the gated-attention `[q | gate]` per-head fused mode (declared,
`NotImplementedError` in the prototype's slicer); the version-keyed table and its packaging; and
anything about cost on a real-sized model.
