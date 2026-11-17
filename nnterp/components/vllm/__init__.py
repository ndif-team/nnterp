"""The base envoys of a vLLM family: the same values as `nnterp.components`, read where vLLM keeps them.

vLLM's own implementations differ from transformers' in three ways a value
has to absorb:

* **No batch axis.** A request's activations are ``[tokens, hidden_size]``,
  served as ``[1, tokens, hidden_size]``, so code written against ``[:, -1]``
  runs on both engines.
* **Live buffers.** A served tensor is the model's own, which the next fused
  kernel rewrites. Every value here is a private copy, handed back to the
  model when the block moves on; in-place edits to the copy and assignment
  both reach the model.
* **The residual stream in two halves.** Most blocks fuse the residual add
  into the next norm and are called with, and return, ``(hidden_states,
  residual)``: the stream is their sum. `FusedLayer` is that block, `Layer`
  one that is called with the stream and returns it.

A family vLLM runs through its transformers backend builds on
`nnterp.components.vllm.transformers_backend` instead.
"""

from .attention import DECODE_STEP, ROTARY, Attention, on_decode_step, rotary_head_size
from .flat import Flat, batched, on_transformers_backend, unbatched
from .layer import FusedLayer, Layer, argument, summed, with_argument
from .mlp import Mlp
from .project import project

__all__ = [
    "Attention", "DECODE_STEP", "Flat", "FusedLayer", "Layer", "Mlp", "ROTARY", "argument", "batched", "on_decode_step",
    "on_transformers_backend", "project", "rotary_head_size", "summed", "unbatched", "with_argument",
]
