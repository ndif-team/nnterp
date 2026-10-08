"""GLM-4.5 / 4.6 (GLM-4-MoE) on vLLM (``vllm.model_executor.models.glm4_moe``).

Llama's names and Llama's fused block: it takes and returns the stream as
``(hidden_states, residual)``. The attention rotates only
``partial_rotary_factor`` of each head and, with ``use_qk_norm``, norms each
head's queries and keys first; both happen before vLLM's attention layer, so
the queries and keys it receives are the ones transformers reads at its
interface. The first ``first_k_dense_replace`` blocks have a dense MLP, the
rest a mixture of experts with a shared expert; either way the ``mlp``'s
output is what the block adds. The routing is inside vLLM's fused MoE kernel
and has no values here.
"""

from vllm.model_executor.models.glm4_moe import Glm4MoE, Glm4MoeAttention, Glm4MoeDecoderLayer, Glm4MoeMLP

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
    """The mixture of experts with its shared expert (or the dense MLP of an early block); its output is what the block adds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Glm4MoeDecoderLayer: Layer, Glm4MoeAttention: Attention, Glm4MoE: Mlp, Glm4MoeMLP: Mlp}
