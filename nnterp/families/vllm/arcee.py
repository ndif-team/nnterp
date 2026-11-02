"""Arcee (AFM) on vLLM (``vllm.model_executor.models.arcee``).

Llama's names and Llama's fused block: it takes and returns the stream as
``(hidden_states, residual)``. The block is vLLM's own class, its attention
is Llama's ``LlamaAttention``, and its MLP is vLLM's ungated one
(``down_proj(relu2(up_proj(x)))``), as on transformers.
"""

from vllm.model_executor.models.arcee import ArceeDecoderLayer, ArceeMLP
from vllm.model_executor.models.llama import LlamaAttention

from ...components.vllm import Attention, FusedLayer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(FusedLayer):
    """The decoder block: ``forward(positions, hidden_states, residual) -> (hidden_states, residual)``."""


class Attention(Attention):
    """Llama's attention; its output is what the block adds, so the base holds."""


class Mlp(Mlp):
    """The ungated MLP; its output is what the block adds, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {ArceeDecoderLayer: Layer, LlamaAttention: Attention, ArceeMLP: Mlp}
