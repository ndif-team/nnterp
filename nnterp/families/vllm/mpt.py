"""MPT on vLLM (``vllm.model_executor.models.mpt``).

Its tree is transformers' (``transformer.{wte, blocks[i].{norm_1, attn,
norm_2, ffn}, norm_f}``); ``lm_head`` is the embedding itself, which the
checkpoint ties. The block is called with the positions and the residual
stream and returns the stream, adding both contributions itself; vLLM's
attention and feed-forward modules return their contributions alone, where
transformers' feed-forward adds the residual inside. The attention is
ALiBi's: no rotary embedding, a per-head slope on the scores.
"""

from vllm.model_executor.models.mpt import MPTAttention, MPTBlock, MPTMLP

from ...components.vllm import Attention, Layer, Mlp
from ..mpt import intermediate_size  # noqa: F401  the size, as the config spells it on transformers

MODEL_TYPES = ("mpt",)

RENAME = {
    "transformer.wte": "embed_tokens",
    "transformer.blocks": "layers",
    "transformer.norm_f": "norm",
    "norm_1": "input_layernorm",
    "attn": "self_attn",
    "norm_2": "post_attention_layernorm",
    "ffn": "mlp",
}


class Layer(Layer):
    """MPT's block: called with the positions and the residual stream, and returns the stream."""

    STREAM = 1


class Attention(Attention):
    """MPT's attention; its output is what the block adds, so the base holds."""


class Mlp(Mlp):
    """MPT's feed-forward; its output is what the block adds, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {MPTBlock: Layer, MPTAttention: Attention, MPTMLP: Mlp}
