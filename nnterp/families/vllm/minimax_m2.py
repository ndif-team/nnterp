"""MiniMax-M2 on vLLM (``vllm.model_executor.models.minimax_m2``).

Llama's names and Llama's fused block, except that the block's feed-forward
is ``block_sparse_moe``, a mixture of experts, aliased to ``mlp``; its output
is what the block adds. The attention norms its queries and keys across the
whole projection before the heads are split and the rotary applied, all
before vLLM's attention layer, so the queries and keys it receives are the
ones transformers reads at its interface. The routing is inside vLLM's fused
MoE kernel and has no values here.
"""

from vllm.model_executor.models.minimax_m2 import MiniMaxM2Attention, MiniMaxM2DecoderLayer, MiniMaxM2MoE

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
ENVOYS = {MiniMaxM2DecoderLayer: Layer, MiniMaxM2Attention: Attention, MiniMaxM2MoE: Mlp}
