"""Gemma 2 (``Gemma2ForCausalLM``).

Llama's names plus a *sandwich* block: every sublayer is normed before and
after, ``x + post_attention_layernorm(attn(input_layernorm(x)))`` and then
``+ post_feedforward_layernorm(mlp(pre_feedforward_layernorm(x)))``. What the
block adds is the post-norm's output, not the module's, so the contributions
point at the sibling norms. ``final_logit_softcapping`` applies to the logits after ``lm_head``.

PaliGemma 2 wraps this text model (its ``text_config.model_type`` is ``gemma2``) as PaliGemma 1
wraps Gemma's: the text stack at ``model.language_model``, a SigLIP tower at
``model.vision_tower``, which is ``vision`` (its ``post_layernorm`` over the patches is
``vision.norm``), and a linear projector ``model.multi_modal_projector``, which is
``projector``; ``vision.image_features`` is read at the scatter (`ImageScatter`). Its processor
refuses a prompt without an image: a text-only trace takes the tokenizer's encoding. The wrapper
never softcaps its logits, whatever its ``text_config`` says, and `project_on_vocab` follows the class.
"""

from typing import TYPE_CHECKING

import torch
from transformers.models.gemma2.modeling_gemma2 import Gemma2Attention, Gemma2DecoderLayer, Gemma2ForCausalLM, Gemma2MLP
from transformers.models.paligemma.modeling_paligemma import PaliGemmaModel
from transformers.models.siglip.modeling_siglip import SiglipAttention, SiglipEncoderLayer, SiglipMLP, SiglipVisionModel

from ..components import Attention, EProperty, ImageScatter, Layer, Mlp, Residual, Vision, VisionAttention, VisionLayer, VisionMlp

if TYPE_CHECKING:
    from ..standardized import StandardizedTransformer

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    # The same text model inside a multimodal wrapper, loaded with task="image-text-to-text":
    # PaliGemmaForConditionalGeneration (PaliGemma 2).
    "model.language_model.embed_tokens": "embed_tokens",
    "model.language_model.layers": "layers",
    "model.language_model.norm": "norm",
    # PaliGemma 2's SigLIP tower and projector. The tower's inner keys are relative to the tower
    # (multi-component, or names no text block has), so they bind on it alone.
    "model.vision_tower": "vision",
    "model.multi_modal_projector": "projector",
    "embeddings.patch_embedding": "patch_embed",
    "encoder.layers": "layers",
    "post_layernorm": "norm",
    "layer_norm1": "input_layernorm",
    "layer_norm2": "post_attention_layernorm",
    # The sublayers' interiors: the output projection, the activation and the down projection under their standard names.
    "out_proj": "o_proj",
    "activation_fn": "act_fn",
    "fc2": "down_proj",
}


class Layer(Layer):
    """Gemma-2's decoder block; returns a bare tensor, so the base holds.

    A sandwich block: ``post_attention_layernorm`` follows the attention, so ``layer_mid`` and
    ``mlp_norm_output`` read ``pre_feedforward_layernorm``, the MLP's pre-norm.
    """

    @EProperty("pre_feedforward_layernorm.input", description="The residual stream after the attention add")
    def layer_mid(self, value) -> Residual:
        return value

    @EProperty("pre_feedforward_layernorm.output", description="The normed stream entering the MLP")
    def mlp_norm_output(self, value) -> Residual:
        return value



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
ENVOYS = {
    Gemma2DecoderLayer: Layer, Gemma2Attention: Attention, Gemma2MLP: Mlp,
    # SigLIP's pre-norm blocks on the shared attention interface: the vision components hold as they are.
    SiglipVisionModel: Vision, SiglipEncoderLayer: VisionLayer, SiglipAttention: VisionAttention, SiglipMLP: VisionMlp,
    PaliGemmaModel: ImageScatter,  # the wrapper's forward scatters the image features: vision.image_features
}


def project_on_vocab(model: "StandardizedTransformer", hidden: torch.Tensor) -> torch.Tensor:
    """The logit lens as the model makes its logits: the final norm, ``lm_head``, and the softcap only where the class applies one.

    ``Gemma2ForCausalLM`` caps with ``final_logit_softcapping``; PaliGemma 2's
    ``PaliGemmaForConditionalGeneration`` never does, though its ``text_config``
    sets 30, so neither does the lens on it.
    """
    logits = model.lm_head(model.norm(hidden))
    cap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
    if cap and isinstance(model._module, Gemma2ForCausalLM):
        return cap * torch.tanh(logits / cap)
    return logits
