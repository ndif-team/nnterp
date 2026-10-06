"""Gemma 3, text (``Gemma3ForCausalLM``, model_type ``gemma3_text``).

Llama's names plus a *sandwich* block: every sublayer is normed before and
after, ``x + post_attention_layernorm(attn(input_layernorm(x)))`` and then
``+ post_feedforward_layernorm(mlp(pre_feedforward_layernorm(x)))``. What the
block adds is the post-norm's output, not the module's, so the contributions
point at the sibling norms. A ``gemma3`` checkpoint (``Gemma3ForConditionalGeneration``,
gemma-3-4b/12b/27b) nests its config's ``text_config``, of this type; the
text-generation task builds the wrapper, whose text stack sits at
``model.language_model.{embed_tokens, layers, norm}`` with ``lm_head`` at the root, so
``RENAME`` carries both spellings and whichever the tree has binds. On the
wrapper the SigLIP tower ``model.vision_tower`` is ``vision`` (a `Vision`; its
``post_layernorm`` over the patches is ``vision.norm``) and the pooling
projector ``model.multi_modal_projector``, whose output is what the wrapper
scatters, is ``projector``. Loaded with ``task="image-text-to-text"`` (the
processor), the tower serves ``vision.image_token_mask`` and ``vision.image_features``. Only
``Gemma3ForCausalLM`` softcaps its logits where the config sets
``final_logit_softcapping`` (no released checkpoint does); the wrapper never
does, and `project_on_vocab` follows the class.
"""

from typing import TYPE_CHECKING

import torch
from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention, Gemma3DecoderLayer, Gemma3ForCausalLM, Gemma3MLP
from transformers.models.siglip.modeling_siglip import SiglipAttention, SiglipEncoderLayer, SiglipMLP, SiglipVisionModel

from ..components import Attention, EProperty, Layer, Mlp, Residual, Vision, VisionAttention, VisionLayer, VisionMlp

if TYPE_CHECKING:
    from ..standardized import StandardizedTransformer

#: The wrappers (config ``model_type``) whose projector's output is what they scatter into the text stream.
IMAGE_WRAPPERS = ("gemma3",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    # A Gemma3ForConditionalGeneration: the same text model under ``model.language_model``.
    "model.language_model.embed_tokens": "embed_tokens",
    "model.language_model.layers": "layers",
    "model.language_model.norm": "norm",
    # The wrapper's SigLIP tower and projector. The tower's inner keys are relative to the
    # tower (multi-component, or names no text block has), so they bind on it alone.
    "model.vision_tower": "vision",
    "model.multi_modal_projector": "projector",
    "embeddings.patch_embedding": "patch_embed",
    "encoder.layers": "layers",
    "post_layernorm": "norm",
    "layer_norm1": "input_layernorm",
    "layer_norm2": "post_attention_layernorm",
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
ENVOYS = {
    Gemma3DecoderLayer: Layer, Gemma3Attention: Attention, Gemma3MLP: Mlp,
    # SigLIP's pre-norm blocks on the shared attention interface: the vision components hold as they are.
    SiglipVisionModel: Vision, SiglipEncoderLayer: VisionLayer, SiglipAttention: VisionAttention, SiglipMLP: VisionMlp,
}


def project_on_vocab(model: "StandardizedTransformer", hidden: torch.Tensor) -> torch.Tensor:
    """The logit lens as the model makes its logits: the final norm, ``lm_head``, and the softcap only where the class applies one.

    ``Gemma3ForCausalLM`` caps with ``final_logit_softcapping`` when the config
    sets it; ``Gemma3ForConditionalGeneration`` never does, whatever its
    ``text_config`` says, so neither does the lens on it.
    """
    logits = model.lm_head(model.norm(hidden))
    cap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
    if cap and isinstance(model._module, Gemma3ForCausalLM):
        return cap * torch.tanh(logits / cap)
    return logits
