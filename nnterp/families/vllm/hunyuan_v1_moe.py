"""Hunyuan MoE V1 (Hunyuan-A13B) on vLLM (``vllm.model_executor.models.hunyuan_v1``).

vLLM runs Hunyuan's dense and MoE models through one implementation. The
names are Llama's and the block is fused, with something more beside the
two halves: it is called ``forward(positions, hidden_states, residual,
kv_states)`` and returns ``(hidden_states, residual, kv_states)``, the stream
being the sum of the first two and the third the keys and values its
attention computed, which the next block reads when the checkpoint shares
them across layers (CLA). So the block's stream is read and written on the
first two elements and the third is passed through, and a skipped block hands
on no keys and values (``None``), so skipping is for checkpoints without
CLA, as are these values: a CLA block's cross-attention, which reuses the
previous block's keys and values, is not mapped.

The attention returns ``(output, (keys, values))``: its contribution is the
first element. It norms its queries and keys per head after the rotary
embedding, as on transformers, and the queries and keys served are the
normed ones the attention layer receives. Every block's ``mlp`` is the
mixture of experts, whose output (the shared expert ``shared_mlp`` included)
is what the block adds; the routing is inside vLLM's fused MoE kernel and has
no values here.
"""

import torch

from vllm.model_executor.models.hunyuan_v1 import HunYuanAttention, HunYuanDecoderLayer, HunYuanSparseMoeBlock

from ...components import EProperty, Residual
from ...components.vllm import Attention, Flat, FusedLayer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "shared_mlp": "shared_experts",
}


class Layer(FusedLayer):
    """The decoder block: ``forward(positions, hidden_states, residual, kv_states) -> (hidden_states, residual, kv_states)``."""

    def skip_with(self, hidden: torch.Tensor) -> None:
        """Skip this block, handing ``hidden`` (``[1, tokens, hidden]``) on as its residual stream, and no keys and values."""
        hidden = hidden.squeeze(0)
        self.skip((hidden, torch.zeros_like(hidden), None))

    @EProperty("output", description="The residual stream leaving the block: hidden_states + residual, [1, tokens, hidden]")
    def layer_output(self, value: tuple) -> Residual:
        return self._stream(value[0], value[1])

    def _with_output(self, output: tuple, view: torch.Tensor) -> tuple:
        hidden, residual, kv_states = output
        return self._edited(hidden, residual, view, f"{self.path}.layer_output"), residual, kv_states

    @layer_output.postprocess
    def layer_output(self, value: torch.Tensor) -> tuple:
        return self._with_output(self.output, value)

    @layer_output.transform
    def layer_output(self, view: torch.Tensor, raw: tuple) -> tuple:
        return self._with_output(raw, view)


class Attention(Attention):
    """The attention: returns ``(output, (keys, values))``, and the output is what the block adds."""

    @Flat("output", select=0, description="What the attention adds to the residual stream, [1, tokens, hidden]")
    def attention_output(self, value) -> Residual:
        return value


class Mlp(Mlp):
    """The mixture of experts, shared expert included; its output is what the block adds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {HunYuanDecoderLayer: Layer, HunYuanAttention: Attention, HunYuanSparseMoeBlock: Mlp}
