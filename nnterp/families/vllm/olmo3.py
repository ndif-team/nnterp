"""OLMo 3 on vLLM (``vllm.model_executor.models.olmo3``).

Llama's names and a plain block: it is called with the positions and the
residual stream and returns the stream. It is post-norm: each sublayer runs
on the stream itself and what the block adds is the sublayer's norm's output,
so the contributions point at the sibling norms, as on transformers. The
attention norms its queries and keys.
"""

from vllm.model_executor.models.olmo3 import Olmo3Attention, Olmo3DecoderLayer, Olmo3MLP

from ...components import Residual
from ...components.vllm import Attention, Flat, Layer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """The decoder block: called with the positions and the stream, post-norm, adding each sublayer's normed output itself."""

    STREAM = 1


class Attention(Attention):
    """The attention: what reaches the residual stream is the post-attention norm's output."""

    @Flat(
        "../post_attention_layernorm.output",
        description="What the attention adds to the residual stream: the post-attention norm's output, [1, tokens, hidden]",
    )
    def attention_output(self, value) -> Residual:
        return value


class Mlp(Mlp):
    """The MLP: what reaches the residual stream is the post-feedforward norm's output."""

    @Flat(
        "../post_feedforward_layernorm.output",
        description="What the MLP adds to the residual stream: the post-feedforward norm's output, [1, tokens, hidden]",
    )
    def mlp_output(self, value) -> Residual:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Olmo3DecoderLayer: Layer, Olmo3Attention: Attention, Olmo3MLP: Mlp}
