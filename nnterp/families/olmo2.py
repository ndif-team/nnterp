"""OLMo 2 (``Olmo2ForCausalLM``).

Llama's names with the norms moved after the sublayers,
``x + post_attention_layernorm(attn(x))`` and then
``+ post_feedforward_layernorm(mlp(x))``. What the
block adds is the post-norm's output, not the module's, so the contributions
point at the sibling norms. OLMo 2 has only the post-norms: no ``input_layernorm`` exists to alias.
"""

from transformers.models.olmo2.modeling_olmo2 import Olmo2Attention, Olmo2DecoderLayer, Olmo2MLP

from ..components import Attention, EProperty, Layer, Mlp, POST_NORM_ATTENTION, POST_NORM_MLP, Residual, unavailable

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """OLMo-2's decoder block; returns a bare tensor, so the base holds.

    A post-norm block: the sublayers read the residual stream itself, so ``layer_mid`` is what the MLP is
    called with and the two pre-norm outputs do not exist.
    """

    attention_norm_output = unavailable(POST_NORM_ATTENTION)

    @EProperty("mlp.input", description="The residual stream after the attention add")
    def layer_mid(self, value) -> Residual:
        return value

    mlp_norm_output = unavailable(POST_NORM_MLP)



class Attention(Attention):
    """OLMo-2's attention: the shared eager forward, but what reaches the residual stream is the post-attention norm's output."""

    @EProperty(
        "../post_attention_layernorm.output",
        description="What the attention adds to the residual stream: the post-attention norm's output",
    )
    def attention_output(self, value) -> Residual:
        return value


class Mlp(Mlp):
    """OLMo-2's MLP: what reaches the residual stream is the post-feedforward norm's output."""

    @EProperty(
        "../post_feedforward_layernorm.output",
        description="What the MLP adds to the residual stream: the post-feedforward norm's output",
    )
    def mlp_output(self, value) -> Residual:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Olmo2DecoderLayer: Layer, Olmo2Attention: Attention, Olmo2MLP: Mlp}
