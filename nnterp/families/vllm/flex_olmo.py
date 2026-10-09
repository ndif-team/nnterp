"""FlexOlmo on vLLM (``vllm.model_executor.models.flex_olmo``).

Llama's names and OLMo-2's post-norm block around a mixture of experts. The
block is not fused, whatever its signature says: it is called
``forward(positions, hidden_states, residual)``, ignores the ``residual`` it is
handed, and returns ``(hidden_states, None)`` with the whole stream first.
The ``None`` is what tells the model's final norm there is no residual to add.
What the block adds is each sublayer's post-norm's output, as on
transformers, so the contributions point at the sibling norms. The attention
norms its queries and keys over all heads at once before the rotary
embedding. Every block's ``mlp`` is the mixture of experts; the routing is
inside vLLM's fused MoE kernel and has no values here.
"""

import torch

from vllm.model_executor.models.flex_olmo import FlexOlmoAttention, FlexOlmoDecoderLayer, FlexOlmoMoE

from ...components import Residual
from ...components.vllm import Attention, Flat, Layer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """The decoder block: called with the positions and the stream, returns ``(stream, None)``."""

    STREAM = 1
    returns_tuple = True

    def skip_with(self, hidden: torch.Tensor) -> None:
        """Skip this block, handing ``hidden`` (``[1, tokens, hidden]``) on as its residual stream, with the ``None`` the block returns beside it."""
        self.skip((hidden.squeeze(0), None))


class Attention(Attention):
    """The attention: what reaches the residual stream is the post-attention norm's output."""

    @Flat(
        "../post_attention_layernorm.output",
        description="What the attention adds to the residual stream: the post-attention norm's output, [1, tokens, hidden]",
    )
    def attention_output(self, value) -> Residual:
        return value


class Mlp(Mlp):
    """The mixture of experts: what reaches the residual stream is the post-feedforward norm's output."""

    @Flat(
        "../post_feedforward_layernorm.output",
        description="What the MLP adds to the residual stream: the post-feedforward norm's output, [1, tokens, hidden]",
    )
    def mlp_output(self, value) -> Residual:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {FlexOlmoDecoderLayer: Layer, FlexOlmoAttention: Attention, FlexOlmoMoE: Mlp}
