"""Qwen3-MoE on vLLM (``vllm.model_executor.models.qwen3_moe``).

Llama's names and Llama's fused block; the attention norms its queries and
keys. A block's ``mlp`` is the sparse mixture of experts, or a dense MLP on
the blocks ``mlp_only_layers`` names; either way its output is what the
block adds. The routing is inside vLLM's fused MoE kernel and has no values
here.
"""

from vllm.model_executor.models.qwen3_moe import Qwen3MoeAttention, Qwen3MoeDecoderLayer, Qwen3MoeMLP, Qwen3MoeSparseMoeBlock

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
    """The mixture of experts (or the dense MLP of an ``mlp_only`` block); its output is what the block adds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Qwen3MoeDecoderLayer: Layer, Qwen3MoeAttention: Attention, Qwen3MoeMLP: Mlp, Qwen3MoeSparseMoeBlock: Mlp}
