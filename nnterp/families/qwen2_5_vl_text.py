"""Qwen2.5-VL, text (``Qwen2_5_VLTextModel``, model_type ``qwen2_5_vl_text``).

The language model inside ``Qwen2_5_VLForConditionalGeneration`` (model_type
``qwen2_5_vl``): Qwen2's block under multimodal rotary embeddings (M-RoPE),
folded into one ``cos``/``sin`` pair before any block runs and applied inside
the attention before the shared interface, so the base `Attention` holds, as on
``qwen2_vl_text``. Every checkpoint loads as the wrapper
(``task="image-text-to-text"``), whose text stack sits at ``model.language_model``.

The wrapper's Qwen ViT ``model.visual`` is ``vision`` (a `QwenVision`: packed,
``[1, patches, vision_hidden]``, and in *window order*: the tower permutes the
patches into attention windows at entry and restores the order after the
merger, so the blocks' values and ``projector.output`` are in window order while
``vision.image_features``, read at the tower's output, is in the scatter's).
"""

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModel, Qwen2_5_VisionTransformerPretrainedModel, Qwen2_5_VLAttention, Qwen2_5_VLDecoderLayer, Qwen2_5_VLMLP,
    Qwen2_5_VLVisionAttention, Qwen2_5_VLVisionBlock, Qwen2MLP,
)

from ..components import ImageScatter, Attention, Layer, Mlp, PackedVisionAttention, PackedVisionLayer, PackedVisionMlp, QwenVision

RENAME = {
    "model.language_model.embed_tokens": "embed_tokens",
    "model.language_model.layers": "layers",
    "model.language_model.norm": "norm",
    # The Qwen ViT and its merger. The tower's inner keys are names no text block has.
    "model.visual": "vision",
    "model.visual.merger": "projector",
    "blocks": "layers",
    "attn": "self_attn",
    "norm1": "input_layernorm",
    "norm2": "post_attention_layernorm",
}


class Layer(Layer):
    """Qwen2.5-VL's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Qwen2.5-VL's attention: M-RoPE is folded into ``cos``/``sin`` before the block and applied before the shared interface, so the base holds."""


class Mlp(Mlp):
    """Qwen2.5-VL's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {
    Qwen2_5_VLModel: ImageScatter,  # the wrapper's forward scatters the image features: vision.image_features
    Qwen2_5_VLDecoderLayer: Layer, Qwen2_5_VLAttention: Attention, Qwen2MLP: Mlp,
    # The packed Qwen ViT: one attention call per window, so the interior is unavailable.
    Qwen2_5_VisionTransformerPretrainedModel: QwenVision, Qwen2_5_VLVisionBlock: PackedVisionLayer,
    Qwen2_5_VLVisionAttention: PackedVisionAttention, Qwen2_5_VLMLP: PackedVisionMlp,
}
