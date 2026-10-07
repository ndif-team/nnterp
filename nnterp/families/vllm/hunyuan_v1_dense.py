"""Hunyuan dense V1 on vLLM (``vllm.model_executor.models.hunyuan_v1``).

Llama's names and a fused block, with a third element: it is called
``forward(positions, hidden_states, residual, kv_states)`` and returns
``(hidden_states, residual, kv_states)``, where the first two are the halves
of the stream and the third the attention's keys and values, handed to the
next block for cross-layer attention sharing (``use_cla``). The attention
returns ``(output, (keys, values))``; the output is what the block adds. It
norms its queries and keys per head after the rotary embedding
(``use_qk_norm``), and the attention layer receives them after both, as on
transformers.

On a checkpoint with ``use_cla`` the blocks that reuse an earlier block's keys
and values run another attention module (``HunYuanCrossAttention``), which
this family does not map; the released dense checkpoints do not set it.
"""

import torch

from vllm.model_executor.models.hunyuan_v1 import HunYuanAttention, HunYuanDecoderLayer, HunYuanMLP

from ...components import EProperty, Residual
from ...components.vllm import Attention, Flat, FusedLayer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(FusedLayer):
    """The decoder block: ``forward(positions, hidden_states, residual, kv_states) -> (hidden_states, residual, kv_states)``."""

    def skip_with(self, hidden: torch.Tensor) -> None:
        """Skip this block, handing ``hidden`` (``[1, tokens, hidden]``) on as its residual stream, and no keys and values."""
        hidden = hidden.squeeze(0)
        self.skip((hidden, torch.zeros_like(hidden), None))

    def _with_output(self, output: tuple, view: torch.Tensor) -> tuple:
        hidden, residual, kv_states = output
        return self._edited(hidden, residual, view, f"{self.path}.layer_output"), residual, kv_states

    @EProperty("output", description="The residual stream leaving the block: hidden_states + residual, [1, tokens, hidden]")
    def layer_output(self, value: tuple) -> Residual:
        return self._stream(value[0], value[1])

    @layer_output.postprocess
    def layer_output(self, value: torch.Tensor) -> tuple:
        return self._with_output(self.output, value)

    @layer_output.transform
    def layer_output(self, view: torch.Tensor, raw: tuple) -> tuple:
        return self._with_output(raw, view)


class Attention(Attention):
    """The attention, which returns ``(output, (keys, values))``; the output is what the block adds."""

    @Flat("output", select=0, description="What the attention adds to the residual stream, [1, tokens, hidden]")
    def attention_output(self, value) -> Residual:
        return value


class Mlp(Mlp):
    """The MLP; its output is what the block adds, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {HunYuanDecoderLayer: Layer, HunYuanAttention: Attention, HunYuanMLP: Mlp}
