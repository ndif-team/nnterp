"""FlexOlmo (``FlexOlmoForCausalLM``).

OLMo-2's post-norm block with a mixture of experts for the MLP:
``x + post_attention_layernorm(attn(x))`` and then
``+ post_feedforward_layernorm(mlp(x))``. What the block adds is the post-norm's
output, not the module's, so the contributions point at the sibling norms. Only
the post-norms exist: no ``input_layernorm`` to alias, and the MLP takes the
residual stream after the attention add. The mixture (``FlexOlmoSparseMoeBlock``)
is on every block and returns the routed experts' sum as one tensor; its experts
are ``intermediate_size`` wide.
"""

from transformers.models.flex_olmo.modeling_flex_olmo import FlexOlmoAttention, FlexOlmoDecoderLayer, FlexOlmoSparseMoeBlock

from ..components import Attention, EProperty, Layer, Moe, Residual

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "gate": "router",
}


class Layer(Layer):
    """FlexOlmo's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """FlexOlmo's attention: the shared eager forward, but what reaches the residual stream is the post-attention norm's output."""

    @EProperty(
        "../post_attention_layernorm.output",
        description="What the attention adds to the residual stream: the post-attention norm's output",
    )
    def attention_output(self, value) -> Residual:
        return value


class Mlp(Moe):
    """FlexOlmo's mixture of experts: what reaches the residual stream is the post-feedforward norm's output."""

    @EProperty(
        "../post_feedforward_layernorm.output",
        description="What the MLP adds to the residual stream: the post-feedforward norm's output",
    )
    def mlp_output(self, value) -> Residual:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {FlexOlmoDecoderLayer: Layer, FlexOlmoAttention: Attention, FlexOlmoSparseMoeBlock: Mlp}
