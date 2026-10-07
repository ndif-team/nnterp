---
title: Contributing
one_liner: House style for nnterp code and docs, the workflow for a change, and the open items nnterp deliberately lacks or has not finished, distilled from GAPS.md.
tags: [developing, contributing, style, workflow, roadmap]
related: [docs/developing/testing.md, docs/developing/architecture.md, docs/developing/transformers-compat.md, docs/developing/gotchas.md, docs/extending/index.md]
sources: [GAPS.md, README.md, nnterp/families/llama.py, nnterp/families/gemma2.py, nnterp/components/eproperty.py, nnterp/components/layer.py, nnterp/components/attention.py, nnterp/components/linear_attention.py, nnterp/standardized.py, tests/families/suite.py, nnsight STYLE.md]
---

# Contributing

## What this is for

How a change lands in nnterp: the style the code and docs are held to, the
steps from a branch to a passing suite, and the list of what nnterp does not
do yet so a contribution can pick one up rather than rediscover it.

## Canonical pattern

A new family is one module and one test file. The module for a Llama-named
checkpoint is three container keys, three empty subclasses and the type
keys (`nnterp/families/llama.py`); a family whose block differs overrides
only the value that differs (`nnterp/families/gemma2.py:33-49`):

```python
"""<Model> (``<Model>ForCausalLM``).

One paragraph: the tree, what the block does with the residual, what the
attention path is, and anything a value has to be pointed at.
"""

from transformers.models.<mt>.modeling_<mt> import <Mt>Attention, <Mt>DecoderLayer, <Mt>MLP

from ..components import Attention, Layer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """<Model>'s decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """<Model>'s attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """<Model>'s MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {<Mt>DecoderLayer: Layer, <Mt>Attention: Attention, <Mt>MLP: Mlp}
```

```python
"""<Model>, end to end."""

from suite import FamilySuite, LLAMA_ROWS

from nnterp.families import <model_type>


class Test<Model>(FamilySuite):
    REPO = "<org>/tiny-random-<Model>ForCausalLM"
    FAMILY = <model_type>
    NATIVE = LLAMA_ROWS
```

Then `HF_HUB_OFFLINE=1 pytest tests/families/test_<model_type>.py -q`.

## House style

nnsight's `STYLE.md` applies to nnterp as written, and its first rules are
the ones that matter most here:

- **Present tense.** The source and the docs describe what is; no "used
  to", no "fixed", no issue numbers, no TODO in prose. When code exists
  because something once broke, write the constraint, not the incident
  (`nnterp/components/eproperty.py:116-118` says why a predicate's
  `AttributeError` becomes a `RuntimeError`, not when it bit).
- **Docstrings state the contract; comments say why.** A module docstring
  teaches the family or the concept (`nnterp/families/falcon.py:1-11`,
  `nnterp/components/recurrent.py:1-14`); a value's docstring says
  what the tensor is, its layout, and how writes behave; an inline comment
  names the failure mode averted or the alternative rejected. A comment that
  paraphrases the line under it is deleted.
- **One word, one meaning.** Use the brief's vocabulary: *family*, *standard
  name*, *standard value*, *contribution*, *available/unavailable*,
  *source-located value*, *interface*, *hybrid*.
- **Descriptions on eproperties are capitalized sentences** without a
  trailing period, naming the layout: `"The attention pattern the values are
  mixed with, [batch, heads, query, key]"`. They are what `repr(model)` and
  `support()` show.
- **One family per module, named after `model_type`**
  (`gemma3_text.py` covers `gemma3_text`); the file name is its type, declared nowhere else.
  The module *is* the registry entry (`nnterp/families/__init__.py:17-22`).
- **Families import their modeling module only inside the family module**,
  at the top of it, never from `nnterp/components` or `nnterp/standardized.py`,
  so `import nnterp` loads no modeling code (`tests/test_registry.py:23-36`).
- **Every value is annotated with a layout name** (`-> Residual`,
  `-> Pattern`, `-> Keys`), never an inline `Float[Tensor, "..."]`. Each name
  is defined in the file of the envoy that serves it (`Residual` in
  `components/layer.py`, the attention interior's in `components/attention.py`,
  the DeltaNet ones in `components/linear_attention.py`, the recurrent state's in
  `components/recurrent.py`, the root's in
  `standardized.py`) and re-exported from `nnterp.components`, which is where a
  family imports it in its one components import line. A family that
  redefines a value writes the base's name, so `layout` and `dims` cannot
  drift from the base, and `test_values_match_their_annotations` checks every
  axis against the model's sizes. A new shape is a new name defined beside the
  envoy that serves it, with a comment above it, and exported from
  `nnterp.components`. Axis names are the shared set in those comments and in
  `suite.py:448-454`.
- **Every source-located value has an `unavailable=` predicate** (`needs_eager`,
  `interface_reason`, `needs_torch_kernels`, or a family's own), so `support()`
  can answer before a trace runs and a read raises `Unavailable` rather than
  `SourceNotAvailable` when the reason is known in advance.
- **A missing value is declared, not omitted**: `attention_probabilities =
  unavailable("...")` keeps the name in the tree with its reason.
- **Tests: one method per value in the suite, one file per family.** A new
  value gets a method in `tests/families/suite.py` that reads it, checks its
  layout, and writes it; a new family gets `tests/families/test_<model_type>.py`
  with a pinned tiny checkpoint and only the tests specific to it.
- **Modern typing**: `from __future__ import annotations`, `X | None`,
  `list[str]`; raise by default, warn only when the program can proceed.

## Workflow

1. Branch from `main`; run git from the repository root and confirm
   `git rev-parse --show-toplevel` prints your nnterp checkout before
   staging ([gotchas.md](gotchas.md)).
2. Make the change: a family module, a component value, or both.
3. Run the family's file, then the whole suite
   (`HF_HUB_OFFLINE=1 pytest -q`, about 71 s; [testing.md](testing.md)).
4. For a new family, add `tests/families/test_<model_type>.py` with a tiny
   checkpoint that is offline-cached; prefer `hf-internal-testing/`,
   `trl-internal-testing/`, `yujiepan/` or `hf-tiny-v2/` repos. If the
   checkpoint's config does not parse on the transformers nnterp is developed on, patch
   the config in the test file the way `tests/families/test_olmo3.py:14-32`
   does.
5. For a new value, add a method to `FamilySuite`, and expect it to run on
   every family: a value one family lacks is declared `unavailable(...)` in
   that family and listed in its `EXPECTED_UNAVAILABLE`.
6. Update the docs page the change touches; every snippet in `docs/` has
   run against a pinned checkpoint before it is published.
7. Commit as `area: lowercase imperative summary` with a body that says why.

## Open items

Distilled from `GAPS.md` (sections 2 and 4, a read-only comparison of
nnterp 1.x on branch `internals-accessors` at `2db11d8` against this
package), which is the source for everything in this list. Two kinds: things nnterp
deliberately does not do the way nnterp 1.x does, and things that remain open.

### Deliberately absent (design choices, from GAPS.md section 3 and 2a)

- **No Llama-like fallback for an unregistered `model_type`.** nnterp 1.x applies
  global name lists to any model; nnterp refuses with `UnsupportedFamily`
  and needs a module even for a family that spells everything like Llama.
  It never mis-binds silently, at the cost of a three-line module per
  family (`GAPS.md` §2a row 1, §3 item 4).
- **Accessors live on the block envoy**, `model.layers[i].layer_output`, not
  on the model as `model.layers_output[i]`; they exist inside a trace and
  appear in the repr (§3 item 1).
- **Containers are lifted to the root**, `model.layers` not `model.model.layers`;
  the final norm keeps Llama's `norm`, not `ln_final` (§3 items 2-3).
- **Eager attention is not forced at load**; `attention_probabilities` says
  `eager` is needed when the checkpoint runs `sdpa` (§3 item 5;
  `tests/test_registry.py:88-92`).
- **Nothing runs at construction** (no scan or trace validation); the suite
  carries the checks (§3 item 8).
- **Remote runs need nnterp installed on the server**; nothing is shipped by
  value, since an `EProperty` cannot be pickled (§2a remote row, §3 item 10).

### Open (from GAPS.md sections 2 and 4)

- **Forward-order ranking metadata** (nnterp 1.x's `Address.order` /
  `Internals.rank`): a way to sort several reads into the order the forward
  reaches them, so a user does not learn Falcon's values-before-queries rule
  by an `OutOfOrderError` (§2a "Forward-order").
- **Load-time validation**: a causal check that a written pattern moves the
  logits, a refusal of heterogeneous layer classes (Mllama's cross-attention
  blocks would be plain `Envoy`s with no `layer_output`), detection of
  `reorder_and_upcast_attn` and of a sublayer that takes a `residual`
  argument (§2a "Load-time validation", §4 items 9-11).
- **Display conveniences**: nnterp 1.x's `plot_topk_tokens`, `prompts_to_df`
  (§2b).
- **VLMs**: `detect_automodel`, `text_only`, `StandardizedVLM`, `load_model`;
  nnterp hardcodes `task="text-generation"` and would need the same container
  lift for a `language_model` (§2a, §3 item 2).
- **Real NDIF verification** of `remote=True` against a deployed
  `TransformersModel` (§2a remote row; the key is right by
  `tests/test_registry.py:80-85`, the round trip is not exercised).
- **Optimized-kernel pinning** for hybrids: with `flash-linear-attention` or
  `causal-conv1d` installed every DeltaNet value is unavailable; nnterp 1.x pins
  the reference kernels at load and re-pins on `dispatch()` (§2a "Hybrid
  Gated DeltaNet", §4 item 6).
- **A `model.require(...)` / `model.available(...)` helper** that answers
  for a set of values at once, instead of reading `support()` by hand and
  meeting `Unavailable` from `hasattr` (§2a "Availability before any
  trace"; the note under `Unavailable` in `nnterp/components/eproperty.py:28-32`).
- **Sequence-first normalization of softmax q/k/v**: the sequence axis is 2
  on `attention_queries`/`keys`/`values` and 1 everywhere else, the layout
  transformers hands its interface; a `seq_axis`-style layout fact or a view
  would make every value sequence-first (§2a "Layout facts", README
  "Layouts").
- **Block interior beyond attention**: nnterp 1.x's `layers_mid`,
  `attentions_premix`, `mlps_activation`, `mlps_neurons` rows have no nnterp
  value yet (§2a "Block accessors").
- **`device_map="auto"` default**, `linear_attention_layers` /
  `attention_layers` index lists, `block_structure` as a family constant
  (§2a, marked trivial).

## Gotchas

- Do not add version conditionals to a family module; the upgrade procedure
  in [transformers-compat.md](transformers-compat.md) moves the module.
- A docs page other than `transformers-compat.md` does not mention versions
  or history.
- `route_kernels(..., "torch")` (or `route_delta_rule(..., "recurrent")`) in a test is
  process-wide: restore `"default"` (`"chunked"`) in a `finally`.
- A tiny checkpoint can be degenerate (DBRX's fp16 weights make every
  pattern uniform); `LOAD_KWARGS` and a family-specific override are the
  tools, not a weaker suite assertion.

## Related

- [testing.md](testing.md)
- [architecture.md](architecture.md) — which layer a change belongs in
- [transformers-compat.md](transformers-compat.md)
- [gotchas.md](gotchas.md)
- `GAPS.md` at the repository root — the full comparison this page distills
- nnsight `STYLE.md`, `docs/developing/contributing.md`
