# Developing

- [Adding a family](adding-a-family.md): One module and one test file for a new model_type.
- [Overriding a value](overriding-values.md): Redefining a standard value when the base does not hold.
- [Custom values](custom-values.md): Adding a value of your own through envoys=.
- [Finding source ops](finding-source-ops.md): Discovering the operation names a source path needs.
- [Registering](registering.md): Registering a family from outside nnterp, or for one load.
- [Architecture](internals/architecture.md): How a model_type becomes a renamed tree with values, and which layer owns what.
- [Descriptors](internals/descriptors.md): How EProperty resolves a path at run time.
- [Recurrent mixers](internals/recurrent-mixers.md): How RecurrentMixer reaches a kernel's values and state.
- [Vision](internals/vision.md): How the vision side of a wrapper is standardized.
- [Testing](testing.md): Running the suite, and what FamilySuite asserts.
- [transformers compatibility](transformers-compatibility.md): What a transformers release can move, and the upgrade procedure.
- [Contributing](contributing.md): House style, the workflow for a change, and the open items.
