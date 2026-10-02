"""Gemma 2 (``Gemma2ForCausalLM``).

Llama's names plus a *sandwich* block: every sublayer is normed before and
after, ``x + post_attention_layernorm(attn(input_layernorm(x)))`` and then
``+ post_feedforward_layernorm(mlp(pre_feedforward_layernorm(x)))``. What the
block adds is the post-norm's output, not the module's, so the contributions
point at the sibling norms. ``final_logit_softcapping`` applies to the logits after ``lm_head``.
"""

from transformers.models.gemma2.modeling_gemma2 import Gemma2Attention, Gemma2DecoderLayer, Gemma2MLP

from ..components import Attention, EProperty, Layer, Mlp, Residual

MODEL_TYPES = ("gemma2",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """Gemma-2's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Gemma-2's attention: the shared eager forward, but what reaches the residual stream is the post-attention norm's output."""

    @EProperty(
        "../post_attention_layernorm.output",
        description="What the attention adds to the residual stream: the post-attention norm's output",
    )
    def attention_output(self, value) -> Residual:
        return value


class Mlp(Mlp):
    """Gemma-2's MLP: what reaches the residual stream is the post-feedforward norm's output."""

    @EProperty(
        "../post_feedforward_layernorm.output",
        description="What the MLP adds to the residual stream: the post-feedforward norm's output",
    )
    def mlp_output(self, value) -> Residual:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Gemma2DecoderLayer: Layer, Gemma2Attention: Attention, Gemma2MLP: Mlp}
