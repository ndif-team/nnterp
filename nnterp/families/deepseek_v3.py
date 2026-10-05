"""DeepSeek-V3 (``DeepseekV3ForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward. Multi-head latent attention as in V2; dense MLPs for the first ``first_k_dense_replace`` blocks, a mixture of experts after.
Moonshot's Kimi K2.5 / K2.6 / K2.7-Code (``Kimi_K25ForConditionalGeneration``) wrap this text
model under ``model.language_model`` with ``lm_head`` at the root, so ``RENAME`` carries both
spellings; their ``text_config`` says ``kimi_k2``, which ``kimi_k2.py`` maps to this family.
"""

from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Attention, DeepseekV3DecoderLayer, DeepseekV3MLP, DeepseekV3MoE

from ..components import Attention, Layer, Mlp, Moe
from .deepseek_v2 import head_dim, qk_head_dim  # noqa: F401  the same latent attention: the same sizes

MODEL_TYPES = ("deepseek_v3",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    # A Kimi_K25ForConditionalGeneration (Kimi K2.5, K2.6, K2.7-Code): the same text model under ``model.language_model``.
    "model.language_model.embed_tokens": "embed_tokens",
    "model.language_model.layers": "layers",
    "model.language_model.norm": "norm",
    "gate": "router",
}


class Layer(Layer):
    """DeepSeek-V3's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """DeepSeek-V3's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """DeepSeek's dense MLP or mixture of experts; both return the hidden states, and the residual is added in the block."""


class Moe(Moe, Mlp):
    """DeepSeek-V3's mixture of experts: a sigmoid router with a selection bias and group-limited top-k, routed experts and shared experts."""

    SCORING = "sigmoid"


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {DeepseekV3DecoderLayer: Layer, DeepseekV3Attention: Attention, DeepseekV3MLP: Mlp, DeepseekV3MoE: Moe}
