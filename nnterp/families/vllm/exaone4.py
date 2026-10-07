"""EXAONE 4.0 on vLLM (``vllm.model_executor.models.exaone4``).

Llama's names, but the block is not fused, whatever its signature says: it
is called ``forward(positions, hidden_states, residual)``, ignores the
``residual`` it is handed, and returns ``(hidden_states, residual)`` with
the *whole* stream first (the second element is the stream after the
attention). It is post-norm: each sublayer runs on the stream itself and what
the block adds is the sublayer's norm's output, so the contributions point at
the sibling norms, as on transformers. The attention norms its queries and
keys, and rotates them only on the blocks the checkpoint's layer types say.
"""

from vllm.model_executor.models.exaone4 import Exaone4Attention, Exaone4DecoderLayer, Exaone4GatedMLP

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
    returns_tuple = True


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
ENVOYS = {Exaone4DecoderLayer: Layer, Exaone4Attention: Attention, Exaone4GatedMLP: Mlp}
