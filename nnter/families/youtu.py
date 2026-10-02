"""Youtu-LLM (``YoutuForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, which returns a bare tensor. Attention is DeepSeek-V2's
multi-head latent attention through the shared eager forward: queries and keys are
``qk_head_dim`` wide (``qk_nope_head_dim + qk_rope_head_dim``), values
``v_head_dim``, and the interface sees ``num_heads`` key/value heads. The config
maps ``head_dim`` to ``qk_rope_head_dim``, the rotary part only, so both widths come
from ``deepseek_v2``'s size functions. The MLP is dense on every block.
"""

from transformers.models.youtu.modeling_youtu import YoutuAttention, YoutuDecoderLayer, YoutuMLP

from ..components import Attention, Layer, Mlp
from .deepseek_v2 import head_dim, qk_head_dim  # noqa: F401  the same latent attention: the same sizes

MODEL_TYPES = ("youtu",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """Youtu's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Youtu's latent attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """Youtu's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {YoutuDecoderLayer: Layer, YoutuAttention: Attention, YoutuMLP: Mlp}
