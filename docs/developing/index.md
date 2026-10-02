---
title: Developing nnterp
one_liner: Internals reference for contributors: architecture, the descriptors, DeltaNet occurrence arithmetic, the test suite, transformers compatibility, gotchas, house style.
tags: [internals, dev, index]
related: [docs/developing/architecture.md, docs/developing/eproperty-internals.md, docs/developing/testing.md]
sources: []
---

# Developing nnterp

One level below `docs/usage/` and `docs/extending/`: these pages cite `file:line`, describe
data flow, and explain the constraints that shape the code. If you want to *use* a value,
start in `docs/usage/`; if you want to add a family, start in `docs/extending/`.

## The one-paragraph model

`StandardizedTransformer.__init__` reads the checkpoint's config, imports the family module
named after its `model_type`, and hands nnsight the family's `RENAME` (aliases) and `ENVOYS`
(module type → envoy subclass). nnsight builds the envoy tree with those classes, so every
block is a `Layer`, every attention an `Attention`, and each carries `EProperty` descriptors
that serve a standard value over an nnsight location: a module's `.output`, another module's
output named relative to this one, or an operation inside the forward reached through
`.source`. Availability is a predicate on the descriptor, checked on the instance before any
read.

## Pages

- [architecture](architecture.md) — the map, which layer owns what, lazy family import.
- [eproperty-internals](eproperty-internals.md) — `EProperty` and its path grammar (`../`, `source`, `input`/`inputs`/`output`), `_resolve`'s walk, drilling per run, `select`, the `Standard` instrumentation rule, `DerivedEProperty`.
- [recurrent-mixer-internals](recurrent-mixer-internals.md) — `RecurrentMixer` and its DeltaNet subclass: the kernel op the forward's own test picks, the once-per-call record (`KERNEL`, `per_call`), per-token state through occurrence arithmetic, `route_kernels`.
- [testing](testing.md) — `HF_HUB_OFFLINE=1 pytest`, what `FamilySuite` asserts method by method, the root tests.
- [transformers-compat](transformers-compat.md) — the versions nnterp is developed against, which operation names a release can move, the upgrade procedure.
- [gotchas](gotchas.md) — contributor traps, each as constraint and reason.
- [contributing](contributing.md) — house style, workflow, open items.

## Related

- nnsight `docs/developing/` — the interleaver, `.source` instrumentation, eproperty; nnterp sits on those.
