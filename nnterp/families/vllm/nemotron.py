"""Nemotron (Nemotron-4 / Minitron) on vLLM (``vllm.model_executor.models.nemotron``).

Llama's names and Llama's fused block, with vLLM's own classes: the block
takes and returns the stream as ``(hidden_states, residual)``, and its
``NemotronLayerNorm1P`` norms (a LayerNorm whose weight is stored minus one)
take the residual beside the hidden state and add the two, as Llama's RMS
norms do. The rotary embedding covers ``partial_rotary_factor`` of each head
and the MLP has no gate, as on transformers.
"""

from vllm.model_executor.models.nemotron import NemotronAttention, NemotronDecoderLayer, NemotronMLP

from ...components.vllm import Attention, FusedLayer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(FusedLayer):
    """The decoder block: ``forward(positions, hidden_states, residual) -> (hidden_states, residual)``."""


class Attention(Attention):
    """The attention; its output is what the block adds, so the base holds."""


class Mlp(Mlp):
    """The MLP; its output is what the block adds, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {NemotronDecoderLayer: Layer, NemotronAttention: Attention, NemotronMLP: Mlp}
