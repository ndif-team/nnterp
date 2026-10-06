---
title: Extending Index
one_liner: Adding a family, redefining a value a family spells differently, adding values of your own, and registering families from outside nnterp.
tags: [extending, index]
related: [docs/usage/index.md, docs/developing/index.md, docs/reference/families.md]
sources: [nnterp/families/__init__.py, nnterp/components/eproperty.py]
---

# Extending Index

A family is one module under `nnterp/families/` named after its `model_type`; a value is an
`EProperty` on an envoy subclass. Both are plain Python, and both can live outside nnterp.

- [adding-a-family](adding-a-family.md) — `RENAME`, `Layer`/`Attention`/`Mlp` subclasses, `ENVOYS`, a `def <size>(model)` where the config spells a root size its own way, and the one test file that proves it.
- [overriding-values](overriding-values.md) — when the base does not hold: an `EProperty` keyed on a path (`../norm.output`, `source.<op>.input`, `source.<call>.inputs` with `select`), `unavailable` markers and predicates, `off_interface`, `seq_first`, a clone with a transform; a root size is a function in the family module, not a descriptor.
- [custom-values](custom-values.md) — a new value on a block, attention or MLP, passed in through `envoys=`; it shows in the repr and in `support()`.
- [finding-source-ops](finding-source-ops.md) — `print(envoy.source)`, `<call>.source` inside a trace, how nnsight names calls and bindings.
- [registering](registering.md) — `nnterp.families.register(module)` for a family outside the package or an override of a shipped one.

## Related

- [docs/developing/eproperty-internals.md](../developing/eproperty-internals.md) — what the descriptors do underneath.
- [docs/developing/testing.md](../developing/testing.md) — what `FamilySuite` asserts of a family.
