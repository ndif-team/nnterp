"""vLLM's families: what vLLM runs for each ``model_type``, standardized.

vLLM runs most architectures through a module of its own, with its own
classes, its own names here and there, and its own forward conventions (see
`nnterp.components.vllm`), so each is a toolkit of its own, named after the
``model_type`` like the transformers ones and declaring the same things:
``RENAME``, ``Layer`` / ``Attention`` / ``Mlp`` and ``ENVOYS``, keyed on the
module types vLLM builds. `nnterp.StandardizedVLLM` looks one up with
``nnterp.families.lookup(model_type, engine="vllm")``.

What a family has to say is which convention its block follows, because the
return type does not tell: `FusedLayer` for a block that takes and returns
``(hidden_states, residual)`` whose sum is the stream, `Layer` for one that
takes and returns the stream, and its own `layer_output` where neither holds.
A family vLLM runs through its transformers backend (transformers' own
modules inside the engine) keys ``ENVOYS`` on transformers' classes and takes
its envoys from `nnterp.components.vllm.transformers_backend`, as they are
where the block is pre-norm with the residual added in the block (`arcee`),
subclassed where it is not (`olmo2`, `phi`).
"""
