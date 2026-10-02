"""OLMoE on vLLM (``vllm.model_executor.models.olmoe``).

Llama's names and Llama's fused block; the attention norms its queries and
keys, and every block's ``mlp`` is a mixture of experts, whose output is what
the block adds. The routing is inside vLLM's fused MoE kernel and has no
values here.
"""

from vllm.model_executor.models.olmoe import OlmoeAttention, OlmoeDecoderLayer, OlmoeMoE

from ...components.vllm import Attention, FusedLayer, Mlp

MODEL_TYPES = ('olmoe',)

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
    """The mixture of experts; its output is what the block adds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {OlmoeDecoderLayer: Layer, OlmoeAttention: Attention, OlmoeMoE: Mlp}
