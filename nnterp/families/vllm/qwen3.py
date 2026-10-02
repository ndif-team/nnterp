"""Qwen3 on vLLM (``vllm.model_executor.models.qwen3``).

Llama's names and Llama's fused block. The attention is its own class (it
norms the queries and keys); the model and the MLP are Qwen2's, which vLLM
reuses.
"""

from vllm.model_executor.models.qwen3 import Qwen3Attention, Qwen3DecoderLayer, Qwen3MLP

from ...components.vllm import Attention, FusedLayer, Mlp

MODEL_TYPES = ("qwen3",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(FusedLayer):
    """Qwen3's decoder block: ``forward(positions, hidden_states, residual) -> (hidden_states, residual)``."""


class Attention(Attention):
    """Qwen3's attention; its output is what the block adds, so the base holds."""


class Mlp(Mlp):
    """Qwen3's MLP (vLLM's ``Qwen2MLP``); its output is what the block adds, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Qwen3DecoderLayer: Layer, Qwen3Attention: Attention, Qwen3MLP: Mlp}
