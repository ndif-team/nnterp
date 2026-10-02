"""Gemma 3, text (``Gemma3ForCausalLM``, model_type ``gemma3_text``).

Llama's names plus a *sandwich* block: every sublayer is normed before and
after, ``x + post_attention_layernorm(attn(input_layernorm(x)))`` and then
``+ post_feedforward_layernorm(mlp(pre_feedforward_layernorm(x)))``. What the
block adds is the post-norm's output, not the module's, so the contributions
point at the sibling norms. A ``gemma3`` checkpoint (``Gemma3ForConditionalGeneration``,
gemma-3-4b/12b/27b) nests its config's ``text_config``, of this type; the
text-generation task builds the wrapper, whose text stack sits at
``model.language_model.{embed_tokens, layers, norm}`` with ``lm_head`` at the root, so
``RENAME`` carries both spellings and whichever the tree has binds.
"""

from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention, Gemma3DecoderLayer, Gemma3MLP

from ..components import Attention, EProperty, Layer, Mlp, Residual

MODEL_TYPES = ("gemma3_text",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    # A Gemma3ForConditionalGeneration: the same text model under ``model.language_model``.
    "model.language_model.embed_tokens": "embed_tokens",
    "model.language_model.layers": "layers",
    "model.language_model.norm": "norm",
}


class Layer(Layer):
    """Gemma-3's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Gemma-3's attention: the shared eager forward, but what reaches the residual stream is the post-attention norm's output."""

    @EProperty(
        "../post_attention_layernorm.output",
        description="What the attention adds to the residual stream: the post-attention norm's output",
    )
    def attention_output(self, value) -> Residual:
        return value


class Mlp(Mlp):
    """Gemma-3's MLP: what reaches the residual stream is the post-feedforward norm's output."""

    @EProperty(
        "../post_feedforward_layernorm.output",
        description="What the MLP adds to the residual stream: the post-feedforward norm's output",
    )
    def mlp_output(self, value) -> Residual:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Gemma3DecoderLayer: Layer, Gemma3Attention: Attention, Gemma3MLP: Mlp}
