"""The base envoys of a vLLM family: the same values as `nnterp.components`, read where vLLM keeps them.

vLLM runs a family through its own implementation, which differs from
transformers' in three ways a value has to absorb:

* **No batch axis.** A request's activations are ``[tokens, hidden]``: every
  prompt token on the prefill, one row on each decode step. The values keep
  nnterp's layouts, so they are served as ``[1, tokens, hidden]`` and code
  written against ``[:, -1]`` runs on both engines.
* **Live buffers.** A served tensor is the model's own, and the next fused
  kernel rewrites it after the block has read it, so a saved reference comes
  back holding later data. Every value here is a private copy, handed back
  to the model when the block moves on; in-place edits to the copy and
  assignment both reach the model.
* **The residual stream in two halves.** Most blocks fuse the residual add
  into the next norm and are called with, and return, ``(hidden_states,
  residual)``: the stream is their sum. `FusedLayer` is that block, `Layer`
  one that is called with the stream (alone, or after the positions) and
  returns it (alone, or with something beside it).

The attention module hands its queries, keys and values to vLLM's attention
layer (its ``attn`` child) and gets the per-head outputs back, each
``[tokens, heads * head_dim]``; they are that child's inputs and output,
served in nnterp's layouts with the heads split out. What the layer computes
between them (the scores, the pattern) is inside its kernel, not Python on
this engine, so those two are recomputed from the queries and keys:
read-only, and there on the prefill only, where a step holds every key.
"""

from .attention import DECODE_STEP, Attention, on_decode_step
from .flat import Flat, batched, unbatched
from .layer import FusedLayer, Layer, argument, with_argument
from .mlp import Mlp
from .project import project

__all__ = [
    "Attention", "DECODE_STEP", "Flat", "FusedLayer", "Layer", "Mlp", "argument", "batched", "on_decode_step", "project", "unbatched",
    "with_argument",
]
