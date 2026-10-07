"""Qwen2-VL, text (``Qwen2VLTextModel``, model_type ``qwen2_vl_text``).

The language model inside ``Qwen2VLForConditionalGeneration`` (model_type
``qwen2_vl``): Qwen2's block under multimodal rotary embeddings (M-RoPE). The
position ids are three streams (temporal, height, width; a fourth, the text
positions, rides in front for the mask), which the model's ``rotary_emb``
folds into one ``cos``/``sin`` pair by ``mrope_section`` before any block
runs. The attention then applies that pair the way Llama does, inside its
forward and before the shared attention interface, so ``attention_queries``
and ``attention_keys`` are the rotated queries and keys, as on every rotary
family, and the base `Attention` holds. No causal-LM class is registered for
the config, so every checkpoint loads as the wrapper (``task="image-text-to-text"``),
whose text stack sits at ``model.language_model``.

The wrapper's Qwen ViT ``model.visual`` is ``vision`` (a `QwenVision`: packed,
its patches ``[1, patches, vision_hidden]``) with its blocks ``vision.layers``;
its ``merger`` is ``projector``, inside the tower.
"""

from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2MLP, Qwen2VLAttention, Qwen2VLDecoderLayer, Qwen2VLModel, Qwen2VLVisionBlock,
    Qwen2VisionTransformerPretrainedModel,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import VisionAttention as Qwen2VLVisionAttention
from transformers.models.qwen2_vl.modeling_qwen2_vl import VisionMlp as Qwen2VLVisionMlp

from ..components import Attention, ImageScatter, Layer, Mlp, QwenVision, QwenVisionAttention, VisionLayer, VisionMlp

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
    """Qwen2-VL's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Qwen2-VL's attention: M-RoPE is folded into ``cos``/``sin`` before the block and applied before the shared interface, so the base holds."""


class Mlp(Mlp):
    """Qwen2-VL's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {
    Qwen2VLModel: ImageScatter,  # the wrapper's forward scatters the image features: vision.image_features
    Qwen2VLDecoderLayer: Layer, Qwen2VLAttention: Attention, Qwen2MLP: Mlp,
    # The Qwen ViT: one attention call per image, so its attention is a QwenVisionAttention.
    Qwen2VisionTransformerPretrainedModel: QwenVision, Qwen2VLVisionBlock: VisionLayer,
    Qwen2VLVisionAttention: QwenVisionAttention, Qwen2VLVisionMlp: VisionMlp,
}
