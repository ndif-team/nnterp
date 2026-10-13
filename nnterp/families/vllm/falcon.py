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

from ...components import PARALLEL, Residual
from ...components.vllm import Attention, Flat, Layer, Mlp
from ..falcon import intermediate_size, num_kv_heads  # noqa: F401  the sizes, as the config spells them on transformers

RENAME = {
    "transformer.word_embeddings": "embed_tokens",
    "transformer.h": "layers",
    "transformer.ln_f": "norm",
    "self_attention": "self_attn",
    # The sublayers' interiors: the output projection, the activation and the down projection under their standard names.
    "dense": "o_proj",
    "act": "act_fn",
    "dense_4h_to_h": "down_proj",
}

BIAS = "the output projection has a bias, which vLLM's block adds apart from the module's output; not mapped yet"


def attention_bias(envoy) -> str | None:
    return BIAS if envoy._module.dense.bias is not None else None


def mlp_bias(envoy) -> str | None:
    return BIAS if envoy._module.dense_4h_to_h.bias is not None else None


def mlp_norm(block) -> str:
    """Where the MLP's normed input is read: its own norm on a sequential block, else the norm the attention shares or ``ln_mlp``."""
    config = block.root.config
    if not config.parallel_attn:
        return "post_attention_layernorm.output"
    return "ln_mlp.output" if getattr(config, "num_ln_in_parallel_attn", None) == 2 else "input_layernorm.output"


class Layer(Layer):
    """Falcon's block: called with the positions and the residual stream, and returns the stream."""

    STREAM = 1

    # The norms a block has depend on the layout: a parallel block (7B) norms the stream once for both sublayers,
    # the 40B layout twice (``ln_attn``, ``ln_mlp``), a sequential one (RW-1B) before each sublayer.

    @Flat(
        lambda block: "ln_attn.output" if getattr(block.root.config, "num_ln_in_parallel_attn", None) == 2 else "input_layernorm.output",
        description="The normed stream entering the attention, [1, tokens, hidden_size]",
    )
    def attention_norm_output(self, value) -> Residual:
        return value

    @Flat(
        "post_attention_layernorm.input",
        unavailable=lambda block: PARALLEL if block.root.config.parallel_attn else None,
        description="The residual stream after the attention add, [1, tokens, hidden_size]",
    )
    def layer_mid(self, value) -> Residual:
        return value

    @Flat(mlp_norm, description="The normed stream entering the MLP, [1, tokens, hidden_size]")
    def mlp_norm_output(self, value) -> Residual:
        return value


class Attention(Attention):
    """Falcon's attention, which returns ``(output, bias)``; the output is what the block adds."""

    @Flat("output", select=0, unavailable=attention_bias, description="What the attention adds to the residual stream, [1, tokens, hidden_size]")
    def attention_output(self, value) -> Residual:
        return value


class Mlp(Mlp):
    """Falcon's MLP, which returns ``(output, bias)``; the output is what the block adds."""

    @Flat("output", select=0, unavailable=mlp_bias, description="What the MLP adds to the residual stream, [1, tokens, hidden_size]")
    def mlp_output(self, value) -> Residual:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {FalconDecoderLayer: Layer, FalconAttention: Attention, FalconMLP: Mlp}
