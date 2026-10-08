"""HyperCLOVA X (``HyperCLOVAXForCausalLM``, NAVER's HyperCLOVA X SEED Think).

Llama's tree with a sandwich block and Granite's µP multipliers::

    h = x + post_norm1(self_attn(input_layernorm(x))) * residual_multiplier
    out = h + post_norm2(mlp(post_attention_layernorm(h))) * residual_multiplier

``post_norm1`` / ``post_norm2`` are RMS norms under ``use_post_norm`` and
identities otherwise; ``post_attention_layernorm`` is the pre-MLP norm (the Llama
meaning). What the block adds is each post-norm's output times the multiplier, so
``attention_output`` and ``mlp_output`` are that product: a computed copy, divided
by the multiplier on assignment, with a transform carrying an in-place edit back
into the post-norm's output, as on Granite; both modules keep the config, where
the multiplier is read. ``attention_multiplier`` is the
softmax scale passed to the shared interface; ``embedding_multiplier`` scales the
embedding module's output before the first block (``token_embeddings`` times it is
``layers[0].input``); ``logits_scaling`` *multiplies* the head's output (Granite
divides), so the family's ``project_on_vocab`` does the same.

HyperCLOVA X Vision V2 (``HyperCLOVAXVisionV2ForConditionalGeneration``, model
type ``hyperclovax_vision_v2``, transformers 5.18 and later) wraps this text model
at ``model.language_model``. Its tower is the Qwen2.5-VL ViT at ``model.vision_model``,
``vision`` (a `QwenVision`: packed, windowed, its own ``merger`` inside), and a linear
``model.projector`` after the tower, ``projector``, maps the merger's output onto
the text width; ``vision.image_features`` is read at the scatter in
``HyperCLOVAXVisionV2Model``'s forward (`ImageScatter`).
"""

from typing import TYPE_CHECKING

import torch
from transformers.models.hyperclovax.modeling_hyperclovax import (
    HyperCLOVAXAttention,
    HyperCLOVAXDecoderLayer,
    HyperCLOVAXMLP,
)

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLMLP, Qwen2_5_VLVisionAttention, Qwen2_5_VLVisionBlock, Qwen2_5_VisionTransformerPretrainedModel,
)

from ..components import (
    Attention, EProperty, ImageScatter, Layer, Mlp, QwenVision, QwenVisionAttention, Residual, VisionLayer, VisionMlp,
)
from .granitemoe import scaled_back

try:  # the vision wrapper arrived in transformers 5.18
    from transformers.models.hyperclovax_vision_v2.modeling_hyperclovax_vision_v2 import HyperCLOVAXVisionV2Model
except ImportError:
    HyperCLOVAXVisionV2Model = None

if TYPE_CHECKING:
    from ..standardized import StandardizedTransformer

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    # The same text model inside HyperCLOVA X Vision V2, loaded with task="image-text-to-text".
    "model.language_model.embed_tokens": "embed_tokens",
    "model.language_model.layers": "layers",
    "model.language_model.norm": "norm",
    # Its Qwen2.5-VL ViT and the linear projector after it. The tower's inner keys are names no text block has.
    "model.vision_model": "vision",
    "model.projector": "projector",
    "blocks": "layers",
    "attn": "self_attn",
    "norm1": "input_layernorm",
    "norm2": "post_attention_layernorm",
}


class Layer(Layer):
    """HyperCLOVA X's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """HyperCLOVA X's attention: the shared eager forward; the block adds its post-norm's output times ``residual_multiplier``."""

    @EProperty("../post_norm1.output", description="What the attention adds to the residual stream: the post-attention norm's output times residual_multiplier")
    def attention_output(self, value) -> Residual:
        return value * self._module.config.residual_multiplier

    @attention_output.postprocess
    def attention_output(self, value):
        return value / self._module.config.residual_multiplier

    @attention_output.transform
    def attention_output(self, value, raw):
        return scaled_back(value, raw, self._module.config.residual_multiplier)


class Mlp(Mlp):
    """HyperCLOVA X's MLP; the block adds its post-norm's output times ``residual_multiplier``."""

    @EProperty("../post_norm2.output", description="What the MLP adds to the residual stream: the post-MLP norm's output times residual_multiplier")
    def mlp_output(self, value) -> Residual:
        return value * self._module.config.residual_multiplier

    @mlp_output.postprocess
    def mlp_output(self, value):
        return value / self._module.config.residual_multiplier

    @mlp_output.transform
    def mlp_output(self, value, raw):
        return scaled_back(value, raw, self._module.config.residual_multiplier)


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {
    HyperCLOVAXDecoderLayer: Layer, HyperCLOVAXAttention: Attention, HyperCLOVAXMLP: Mlp,
    # The Qwen2.5-VL ViT of HyperCLOVA X Vision V2: one attention call per window, so a QwenVisionAttention.
    Qwen2_5_VisionTransformerPretrainedModel: QwenVision, Qwen2_5_VLVisionBlock: VisionLayer,
    Qwen2_5_VLVisionAttention: QwenVisionAttention, Qwen2_5_VLMLP: VisionMlp,
}
if HyperCLOVAXVisionV2Model is not None:
    ENVOYS[HyperCLOVAXVisionV2Model] = ImageScatter  # the wrapper's forward scatters the image features: vision.image_features


def project_on_vocab(model: "StandardizedTransformer", hidden: torch.Tensor) -> torch.Tensor:
    """The logit lens as the model makes its logits: the final norm, ``lm_head``, then times ``logits_scaling``."""
    return model.lm_head(model.norm(hidden)) * model.config.get_text_config().logits_scaling
