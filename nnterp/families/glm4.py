"""GLM-4 (``Glm4ForCausalLM``, GLM-4-0414 and GLM-Z1).

Llama's tree with a sandwich block of four norms::

    h = x + post_self_attn_layernorm(self_attn(input_layernorm(x)))
    out = h + post_mlp_layernorm(mlp(post_attention_layernorm(h)))

What the block adds is each post-norm's output, not the module's, so
``attention_output`` points at ``post_self_attn_layernorm`` and ``mlp_output`` at
``post_mlp_layernorm``. Unlike Gemma-2, ``post_attention_layernorm`` here is the
pre-MLP norm (the Llama meaning), so the standard aliases need no renaming and
``mlp.input`` is its output. The attention runs the shared eager forward with a
partial rotary embedding; the MLP is a fused ``gate_up_proj``.
"""

from transformers.models.glm4.modeling_glm4 import Glm4Attention, Glm4DecoderLayer, Glm4MLP

from ..components import Attention, EProperty, Layer, Mlp, Residual

MODEL_TYPES = ("glm4",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """GLM-4's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """GLM-4's attention: the shared eager forward, but what reaches the residual stream is the post-self-attention norm's output."""

    @EProperty(
        "../post_self_attn_layernorm.output",
        description="What the attention adds to the residual stream: the post-self-attention norm's output",
    )
    def attention_output(self, value) -> Residual:
        return value


class Mlp(Mlp):
    """GLM-4's MLP: what reaches the residual stream is the post-MLP norm's output."""

    @EProperty(
        "../post_mlp_layernorm.output",
        description="What the MLP adds to the residual stream: the post-MLP norm's output",
    )
    def mlp_output(self, value) -> Residual:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Glm4DecoderLayer: Layer, Glm4Attention: Attention, Glm4MLP: Mlp}
