"""vLLM's families: its own implementation of each ``model_type``, standardized.

vLLM does not run transformers' modules. It has a module of its own per
architecture, with its own classes, its own names here and there, and its own
forward conventions (see `nnterp.components.vllm`), so each is a toolkit of its
own, named after the ``model_type`` like the transformers ones and declaring
the same things: ``MODEL_TYPES``, ``RENAME``, ``Layer`` / ``Attention`` /
``Mlp`` and ``ENVOYS``, keyed on vLLM's module types.
`nnterp.StandardizedVLLM` looks one up with
``nnterp.families.lookup(model_type, engine="vllm")``.

What a family has to say is which convention its block follows, because the
return type does not tell: `FusedLayer` for a block that takes and returns
``(hidden_states, residual)`` whose sum is the stream, `Layer` for one that
takes and returns the stream, and its own `layer_output` where neither holds.
"""
