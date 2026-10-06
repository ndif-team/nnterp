"""Gemma 4 unified, text (``Gemma4UnifiedForCausalLM``, model_type ``gemma4_unified_text``): Gemma-4-12B.

Gemma-4's tree and block without per-layer embeddings or a mixture of experts:
the sandwich, then ``out = (x + post_attention_layernorm(...) +
post_feedforward_layernorm(...)) * layer_scalar``, in place. The contributions are the
post-norms' outputs, served unscaled, so the identity is
``(input + attention_output + mlp_output) * layer_scalar == layer_output``. KV
sharing, ``attention_k_eq_v`` and the per-layer ``head_dim`` are Gemma-4's
(`gemma4_text`), and so is the attention envoy (keys and values served as
copies private to the block), and so are the root's ``head_dim`` and
``num_kv_heads``: the config's top-level values. A ``gemma4_unified`` checkpoint
(``Gemma4UnifiedForConditionalGeneration``) keeps the text stack at
``model.language_model``, so ``RENAME`` carries both spellings.
"""

from transformers.models.gemma4_unified.modeling_gemma4_unified import (
    Gemma4UnifiedTextAttention, Gemma4UnifiedTextDecoderLayer, Gemma4UnifiedTextMLP,
)

from ..components import EProperty, Layer, Mlp, Residual
from . import gemma4_text
from .gemma4_text import RENAME, head_dim, num_kv_heads  # noqa: F401  the same tree; the sizes read the same config keys


class Layer(Layer):
    """Gemma-4 unified's decoder block; returns a bare tensor (the sum times ``layer_scalar``), so the base holds."""


class Attention(gemma4_text.Attention):
    """Gemma-4 unified's attention: Gemma-4's, with its post-norm contribution and its keys and values private to the block."""


class Mlp(Mlp):
    """Gemma-4 unified's MLP: what reaches the residual stream is the post-feedforward norm's output."""

    @EProperty(
        "../post_feedforward_layernorm.output",
        description="What the MLP adds to the residual stream: the post-feedforward norm's output",
    )
    def mlp_output(self, value) -> Residual:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Gemma4UnifiedTextDecoderLayer: Layer, Gemma4UnifiedTextAttention: Attention, Gemma4UnifiedTextMLP: Mlp}
