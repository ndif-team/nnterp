"""Falcon on vLLM (``vllm.model_executor.models.falcon``).

Its tree is transformers' (``transformer.{word_embeddings, h[i].{input_layernorm
or ln_attn + ln_mlp, self_attention, mlp}, ln_f}`` plus ``lm_head``). The
block is called with the positions and the residual stream and returns the
stream. Its attention and its MLP each return ``(output, bias)``: vLLM keeps
the output projection's bias apart so that a parallel block can sum the two
outputs before one all-reduce, and adds it in the block. The contributions
are the first element, which is the whole contribution on a checkpoint whose
projections have no bias (Falcon-7B and -40B); on one that has (``bias`` in
the config, Falcon-RW) they are unavailable.

On a parallel block the block adds the attention into the MLP's output
tensor in place; ``mlp_output`` is read before that, as a copy, like every
value on this engine.
"""

from vllm.model_executor.models.falcon import FalconAttention, FalconDecoderLayer, FalconMLP

from ...components import Residual
from ...components.vllm import Attention, Flat, Layer, Mlp
from ..falcon import intermediate_size, num_kv_heads  # noqa: F401  the sizes, as the config spells them on transformers

MODEL_TYPES = ("falcon",)

RENAME = {
    "transformer.word_embeddings": "embed_tokens",
    "transformer.h": "layers",
    "transformer.ln_f": "norm",
    "self_attention": "self_attn",
}

BIAS = "the output projection has a bias, which vLLM's block adds apart from the module's output; not mapped yet"


def attention_bias(envoy) -> str | None:
    return BIAS if envoy._module.dense.bias is not None else None


def mlp_bias(envoy) -> str | None:
    return BIAS if envoy._module.dense_4h_to_h.bias is not None else None


class Layer(Layer):
    """Falcon's block: called with the positions and the residual stream, and returns the stream."""

    STREAM = 1


class Attention(Attention):
    """Falcon's attention, which returns ``(output, bias)``; the output is what the block adds."""

    @Flat("output", select=0, unavailable=attention_bias, description="What the attention adds to the residual stream, [1, tokens, hidden]")
    def attention_output(self, value) -> Residual:
        return value


class Mlp(Mlp):
    """Falcon's MLP, which returns ``(output, bias)``; the output is what the block adds."""

    @Flat("output", select=0, unavailable=mlp_bias, description="What the MLP adds to the residual stream, [1, tokens, hidden]")
    def mlp_output(self, value) -> Residual:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {FalconDecoderLayer: Layer, FalconAttention: Attention, FalconMLP: Mlp}
