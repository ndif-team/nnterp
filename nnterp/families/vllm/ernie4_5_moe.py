"""ERNIE 4.5 MoE on vLLM (``vllm.model_executor.models.ernie45_moe``).

Llama's names and Llama's fused block: it takes and returns the stream as
``(hidden_states, residual)``. A block's ``mlp`` is the mixture of experts on
the blocks from ``moe_layer_start_index`` to ``moe_layer_end_index`` (every
``moe_layer_interval``-th), a dense MLP on the others; either way its output is
what the block adds, the shared experts' included, as on transformers. The
routing is inside vLLM's fused MoE kernel and has no values here.
"""

from vllm.model_executor.models.ernie45_moe import Ernie4_5_MoeAttention, Ernie4_5_MoeDecoderLayer, Ernie4_5_MoeMLP, Ernie4_5_MoeMoE

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
    """The mixture of experts with its shared experts (or the dense MLP of a block outside the MoE range); its output is what the block adds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Ernie4_5_MoeDecoderLayer: Layer, Ernie4_5_MoeAttention: Attention, Ernie4_5_MoeMoE: Mlp, Ernie4_5_MoeMLP: Mlp}
