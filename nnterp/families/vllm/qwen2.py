"""Qwen2 and Qwen2.5 on vLLM (``vllm.model_executor.models.qwen2``).

Llama's names and Llama's fused block, in classes of its own.
"""

from vllm.model_executor.models.qwen2 import Qwen2Attention, Qwen2DecoderLayer, Qwen2MLP

from ...components.vllm import Attention, FusedLayer, Mlp

MODEL_TYPES = ("qwen2",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(FusedLayer):
    """Qwen2's decoder block: ``forward(positions, hidden_states, residual) -> (hidden_states, residual)``."""


class Attention(Attention):
    """Qwen2's attention; its output is what the block adds, so the base holds."""


class Mlp(Mlp):
    """Qwen2's MLP; its output is what the block adds, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Qwen2DecoderLayer: Layer, Qwen2Attention: Attention, Qwen2MLP: Mlp}
