"""GLM-4.7-Flash (GLM-4-MoE-Lite) on vLLM (``vllm.model_executor.models.glm4_moe_lite``).

DeepSeek-V2's block under GLM's names: Llama's names, a fused block that takes
and returns the stream as ``(hidden_states, residual)``, DeepSeek's latent
attention (vLLM's own DeepSeek-V2 attention classes, subclassed) and GLM-4-MoE's
mixture of experts. A block's ``mlp`` is the mixture of experts with its shared
expert, or a dense MLP on the first ``first_k_dense_replace`` blocks; either
way its output is what the block adds, and the routing is inside vLLM's fused
MoE kernel.

The attention is latent (MLA). By default vLLM runs it in a kernel of its own
(``mla_attn``), which works on the compressed keys and values and serves
nothing of the per-head queries, keys and values transformers reads; with
``VLLM_MLA_DISABLE=1`` it runs its ordinary attention layer on padded
tensors. Either way the attention's interior is not mapped here, and its
contribution, which is the module's output on both paths, is.

The two engines' latent norms (``q_a_layernorm``, ``kv_a_layernorm``) differ:
vLLM's take the config's ``rms_norm_eps`` (1e-5 on GLM-4.7-Flash),
transformers' a fixed 1e-6, so the attention's values differ between them by
that much (8e-4 of the queries' scale in the first block of GLM-4.7-Flash).
"""

from vllm.model_executor.models.glm4_moe_lite import (
    Glm4MoeLite, Glm4MoeLiteAttention, Glm4MoeLiteDecoderLayer, Glm4MoeLiteMLAAttention, Glm4MoeLiteMLP,
)

from ...components import unavailable
from ...components.vllm import Attention, FusedLayer, Mlp
from ..glm4_moe_lite import head_dim, qk_head_dim  # noqa: F401  the latent attention's sizes, as on transformers

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}

LATENT = "vLLM runs GLM-4-MoE-Lite's latent attention in its own kernel, on compressed keys and values; not mapped on vLLM"


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
    """The mixture of experts with its shared expert (or the dense MLP of an early block); its output is what the block adds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {
    Glm4MoeLiteDecoderLayer: Layer,
    Glm4MoeLiteMLAAttention: Attention, Glm4MoeLiteAttention: Attention,
    Glm4MoeLite: Mlp, Glm4MoeLiteMLP: Mlp,
}
