"""GLM-5 (``GlmMoeDsaForCausalLM``).

DeepSeek-V3.2's tree and block (``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual added
in the block): multi-head latent attention with DeepSeek Sparse Attention, the
indexer's top-k selection folded into the attention mask before the shared
``attention_interface`` call under eager, so the interior is the base's and means what
it means on ``deepseek_v32``: ``attention_scores`` carry the sparse mask,
``attention_probabilities`` is the dense ``[batch, heads, query, key]`` pattern, zero
outside each query's selected keys. While a prompt is no longer than ``index_topk``
the selection is every causal key.

What differs is that the selection is shared across blocks. ``config.indexer_types``
marks each block ``"full"`` (it has an ``indexer`` and selects its own keys) or
``"shared"`` (no indexer; it reuses the selection of the block before). The selection
therefore travels between blocks: the attention returns
``(attn_output, attn_weights, topk_indices)`` and the block returns
``(hidden_states, topk_indices)``, so ``returns_tuple`` is set; ``attention_output``
and ``layer_output`` take the first element. The selection is
``self_attn.output[2]`` on every block (``self_attn.indexer.output`` on a full one);
it has no standard name. ``skip_layers`` hands a skipped block's successor ``None`` for
the selection, which a shared block refuses (transformers raises ``ValueError``), so
skip up to a full block or to the end. Dense and mixture-of-experts MLPs follow
``mlp_layer_types``. The sizes are DeepSeek-V2's.
"""

from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaAttention, GlmMoeDsaDecoderLayer, GlmMoeDsaMLP, GlmMoeDsaMoE

from ..components import Attention, Layer, Mlp, Moe
from .deepseek_v2 import head_dim, qk_head_dim  # noqa: F401  the same latent attention: the same sizes

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "gate": "router",
}


class Layer(Layer):
    """GLM-5's decoder block; returns ``(hidden_states, topk_indices)``, which the base unwraps."""

    returns_tuple = True


class Attention(Attention):
    """GLM-5's sparse latent attention; the selection is folded into the mask before the shared eager forward, and the base takes the first of the three returns."""


class Mlp(Mlp):
    """GLM-5's dense MLP or mixture of experts; both return the hidden states, and the residual is added in the block."""


class Moe(Moe, Mlp):
    """GLM-5's mixture of experts: DeepSeek-V3's sigmoid router, routed experts and a shared expert."""

    SCORING = "sigmoid"


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {GlmMoeDsaDecoderLayer: Layer, GlmMoeDsaAttention: Attention, GlmMoeDsaMLP: Mlp, GlmMoeDsaMoE: Moe}
