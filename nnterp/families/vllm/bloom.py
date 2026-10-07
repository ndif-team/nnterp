"""BLOOM on vLLM (``vllm.model_executor.models.bloom``).

Its tree is transformers' (``transformer.{word_embeddings,
word_embeddings_layernorm, h[i].{input_layernorm, self_attention,
post_attention_layernorm, mlp}, ln_f}`` plus ``lm_head``). The block is
called with the positions and the residual stream and returns the stream,
adding both contributions itself; vLLM's attention and MLP modules return
their contributions alone, where transformers' add the residual inside. The
attention is ALiBi's: no rotary embedding, a per-head slope on the scores.
"""

from vllm.model_executor.models.bloom import BloomAttention, BloomBlock, BloomMLP

from ...components.vllm import Attention, Layer, Mlp
from ..bloom import intermediate_size  # noqa: F401  the size, as the config spells it on transformers

RENAME = {
    "transformer.word_embeddings": "embed_tokens",
    "transformer.h": "layers",
    "transformer.ln_f": "norm",
    "self_attention": "self_attn",
}


class Layer(Layer):
    """BLOOM's block: called with the positions and the residual stream, and returns the stream."""

    STREAM = 1


class Attention(Attention):
    """BLOOM's attention; its output is what the block adds, so the base holds."""


class Mlp(Mlp):
    """BLOOM's MLP; its output is what the block adds, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {BloomBlock: Layer, BloomAttention: Attention, BloomMLP: Mlp}
