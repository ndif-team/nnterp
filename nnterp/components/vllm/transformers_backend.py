"""The base envoys of a family vLLM runs through its transformers backend: transformers' own modules, inside vLLM.

The block, the attention and the MLP are transformers' classes, called the
way transformers calls them, so the values sit where they sit on
transformers. Three things differ, and these envoys absorb them:

* **The batch axis is there.** Inside the model a request's activations are
  ``[1, tokens, hidden_size]``. The values are served as private copies all
  the same (`Flat` with ``batch=True``).
* **The attention layer is vLLM's.** The attention module's ``attn`` child is
  the engine's attention layer, the one the vLLM `Attention` base reads.
* **Some modules are vLLM's.** The backend fuses the query, key and value
  projections into one ``qkv_proj`` (and a gated MLP's into ``gate_up_proj``),
  rewriting the module's forward, where nnsight's ``.source`` then returns
  the class's text (docs/developing/gotchas.md). So the queries and keys
  before the rotary embedding, arguments of a call inside the attention's
  forward, are not served here (`BACKEND_ROTARY`); the limit lifts when vLLM
  registers the rewritten text in ``linecache`` under its own file name and
  nnsight instruments the instance's forward.
"""

from __future__ import annotations

import torch

from .. import layer as standard
from ..layer import Residual
from .attention import Attention as VLLMAttention
from .layer import Layer as VLLMLayer
from .flat import Flat
from .mlp import Mlp as VLLMMlp


#: Why the queries and keys before the rotary, and `rotate`, are not served on this backend.
BACKEND_ROTARY = (
    "not served on vLLM's transformers backend: the rotary is a call inside the attention's forward, which the backend "
    "rewrites at load and nnsight's .source cannot instrument; read the queries and keys before it, and rotate, on a "
    "StandardizedTransformer"
)


class Layer(standard.Layer):
    """A transformers block on vLLM: called with the stream ``[1, tokens, hidden_size]``, returning it."""

    @Flat("input", batch=True, description="The residual stream entering the block, [1, tokens, hidden_size]")
    def layer_input(self, value: torch.Tensor) -> Residual:
        return value

    @Flat("output", select=lambda envoy: 0 if envoy.returns_tuple else None, batch=True, description="The residual stream leaving the block, [1, tokens, hidden_size]")
    def layer_output(self, value: torch.Tensor) -> Residual:
        return value

    attention_input_gain = VLLMLayer.attention_input_gain
    attention_output_gain = VLLMLayer.attention_output_gain

    #: The boundaries between them, read as on every vLLM block (with the batch axis this backend keeps).
    attention_norm_output = VLLMLayer.attention_norm_output
    layer_mid = VLLMLayer.layer_mid
    mlp_norm_output = VLLMLayer.mlp_norm_output
    mlp_input = VLLMLayer.mlp_input


class Attention(VLLMAttention):
    """A transformers attention module on vLLM, whose ``attn`` child is the engine's attention layer.

    It returns ``(attn_output, attn_weights)``; the first is what the block
    adds, unless a family's block norms it first.
    """

    def no_rotary(self) -> str | None:
        return BACKEND_ROTARY

    @Flat("output", select=0, batch=True, description="What the attention adds to the residual stream, [1, tokens, hidden_size]")
    def attention_output(self, value: torch.Tensor) -> Residual:
        return value


class Mlp(VLLMMlp):
    """A transformers MLP on vLLM: its output is what the block adds, unless a family's block norms it first."""

    @Flat("output", batch=True, description="What the MLP adds to the residual stream, [1, tokens, hidden_size]")
    def mlp_output(self, value: torch.Tensor) -> Residual:
        return value
