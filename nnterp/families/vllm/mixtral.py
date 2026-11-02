"""Mixtral on vLLM (``vllm.model_executor.models.mixtral``).

Llama's names and Llama's fused block, except that the block's feed-forward
is ``block_sparse_moe``, a mixture of experts, aliased to ``mlp``; its output
is what the block adds. The routing is inside vLLM's fused MoE kernel and has
no values here.
"""

from vllm.model_executor.models.mixtral import MixtralAttention, MixtralDecoderLayer, MixtralMoE

from ...components.vllm import Attention, FusedLayer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "block_sparse_moe": "mlp",
}


class Layer(FusedLayer):
    """The decoder block: ``forward(positions, hidden_states, residual) -> (hidden_states, residual)``."""


class Attention(Attention):
    """The attention; its output is what the block adds, so the base holds."""


class Mlp(Mlp):
    """The mixture of experts; its output is what the block adds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {MixtralDecoderLayer: Layer, MixtralAttention: Attention, MixtralMoE: Mlp}
