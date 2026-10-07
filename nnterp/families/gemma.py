"""Gemma 1 (``GemmaForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward.

PaliGemma puts a SigLIP tower at ``model.vision_tower``, which is ``vision`` (its
``post_layernorm`` over the patches is ``vision.norm``), and a linear projector
``model.multi_modal_projector``, which is ``projector``; ``vision.image_features``
is read at the scatter (`ImageScatter`). Its processor refuses a prompt without
an image: a text-only trace takes the tokenizer's encoding.
"""

from transformers.models.gemma.modeling_gemma import GemmaAttention, GemmaDecoderLayer, GemmaMLP
from transformers.models.paligemma.modeling_paligemma import PaliGemmaModel
from transformers.models.siglip.modeling_siglip import SiglipAttention, SiglipEncoderLayer, SiglipMLP, SiglipVisionModel

from ..components import Attention, ImageScatter, Layer, Mlp, Vision, VisionAttention, VisionLayer, VisionMlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    # The same text model inside a multimodal wrapper, loaded with task="image-text-to-text":
    # PaliGemmaForConditionalGeneration.
    "model.language_model.embed_tokens": "embed_tokens",
    "model.language_model.layers": "layers",
    "model.language_model.norm": "norm",
    # PaliGemma's SigLIP tower and projector. The tower's inner keys are relative to the tower
    # (multi-component, or names no text block has), so they bind on it alone.
    "model.vision_tower": "vision",
    "model.multi_modal_projector": "projector",
    "embeddings.patch_embedding": "patch_embed",
    "encoder.layers": "layers",
    "post_layernorm": "norm",
    "layer_norm1": "input_layernorm",
    "layer_norm2": "post_attention_layernorm",
}


class Layer(Layer):
    """Gemma's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Gemma's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """Gemma's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {
    GemmaDecoderLayer: Layer, GemmaAttention: Attention, GemmaMLP: Mlp,
    # SigLIP's pre-norm blocks on the shared attention interface: the vision components hold as they are.
    SiglipVisionModel: Vision, SiglipEncoderLayer: VisionLayer, SiglipAttention: VisionAttention, SiglipMLP: VisionMlp,
    PaliGemmaModel: ImageScatter,  # the wrapper's forward scatters the image features: vision.image_features
}
