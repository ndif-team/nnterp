"""DeepSeek-V3 on vLLM (``vllm.model_executor.models.deepseek_v2``).

vLLM runs DeepSeek-V2 and -V3 through one implementation. The names are
Llama's and the block is fused: it takes and returns the stream as
``(hidden_states, residual)``. A block's ``mlp`` is the mixture of experts,
or a dense MLP on the first ``first_k_dense_replace`` blocks; either way its
output is what the block adds, and the routing is inside vLLM's fused MoE
kernel.

The attention is latent (MLA). By default vLLM runs it in a kernel of its own
(``mla_attn``), which works on the compressed keys and values and serves
nothing of the per-head queries, keys and values transformers reads; with
``VLLM_MLA_DISABLE=1`` it runs its ordinary attention layer on padded
tensors. Either way the attention's interior is not mapped here, and its
contribution, which is the module's output on both paths, is.
"""

from vllm.model_executor.models.deepseek_v2 import (
    DeepseekAttention, DeepseekV2Attention, DeepseekV2DecoderLayer, DeepseekV2MLAAttention, DeepseekV2MLP, DeepseekV2MoE,
)

from ...components import unavailable
from ...components.vllm import Attention, FusedLayer, Mlp
from ..deepseek_v2 import head_dim, qk_head_dim  # noqa: F401  the latent attention's sizes, as on transformers

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}

LATENT = "vLLM runs DeepSeek's latent attention in its own kernel, on compressed keys and values; not mapped on vLLM"


class Layer(FusedLayer):
    """The decoder block: ``forward(positions, hidden_states, residual, llama_4_scaling) -> (hidden_states, residual)``."""


class Attention(Attention):
    """The latent attention: its output is what the block adds; its interior is not mapped."""

    attention_queries = unavailable(LATENT)
    attention_keys = unavailable(LATENT)
    attention_values = unavailable(LATENT)
    attention_scores = unavailable(LATENT)
    attention_probabilities = unavailable(LATENT)
    attention_head_outputs = unavailable(LATENT)


class Mlp(Mlp):
    """The mixture of experts (or the dense MLP of an early block); its output is what the block adds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {
    DeepseekV2DecoderLayer: Layer,
    DeepseekV2MLAAttention: Attention, DeepseekV2Attention: Attention, DeepseekAttention: Attention,
    DeepseekV2MoE: Mlp, DeepseekV2MLP: Mlp,
}
