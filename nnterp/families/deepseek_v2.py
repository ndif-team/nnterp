"""DeepSeek-V2 (``DeepseekV2ForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward. Multi-head latent attention: queries and keys are ``qk_head_dim`` wide (``qk_nope_head_dim + qk_rope_head_dim``), values ``v_head_dim``, and the interface sees ``num_heads`` key/value heads whatever ``num_key_value_heads`` says. The first ``first_k_dense_replace`` blocks have a dense MLP, the rest a mixture of experts.
"""

from typing import TYPE_CHECKING

from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2Attention, DeepseekV2DecoderLayer, DeepseekV2MLP, DeepseekV2Moe

from ..components import Attention, Layer, Mlp, Moe

if TYPE_CHECKING:
    from ..standardized import StandardizedTransformer

MODEL_TYPES = ("deepseek_v2",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "gate": "router",
}


class Layer(Layer):
    """DeepSeek-V2's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """DeepSeek-V2's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """DeepSeek's dense MLP or mixture of experts; both return the hidden states, and the residual is added in the block."""


class Moe(Moe, Mlp):
    """DeepSeek-V2's mixture of experts: a softmax router (greedy or group-limited), routed experts and shared experts."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {DeepseekV2DecoderLayer: Layer, DeepseekV2Attention: Attention, DeepseekV2MLP: Mlp, DeepseekV2Moe: Moe}


# -- sizes: multi-head latent attention gives queries and keys their own width ------

def head_dim(model: "StandardizedTransformer") -> int:
    """Width of one head's values and outputs: ``v_head_dim`` (the config's ``head_dim`` is the latent width, which no served value has)."""
    return model.config.get_text_config().v_head_dim


def qk_head_dim(model: "StandardizedTransformer") -> int:
    """Width of one head's queries and keys: the non-rotary part plus the rotary part."""
    config = model.config.get_text_config()  # a multimodal wrapper's (Kimi K2.5) nests it
    return config.qk_nope_head_dim + config.qk_rope_head_dim
