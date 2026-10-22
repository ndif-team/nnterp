"""Ministral 3 (``Ministral3ForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward.
The queries are scaled by position (``llama_4_scaling_beta``) after the rotary
embedding and before the interface, so ``attention_queries`` are the scaled ones.

The ``mistral3`` wrapper puts a Pixtral tower at ``model.vision_tower``, which is
``vision`` (a `PixtralVision`: every image's patches packed in one row, no final
norm), and the projector ``model.multi_modal_projector`` (a norm, the
``patch_merger`` that merges each 2x2 patch block, two linears), which is
``projector``; ``vision.image_features`` is read at the scatter (`ImageScatter`).
"""

from transformers.models.ministral3.modeling_ministral3 import (
    Ministral3Attention,
    Ministral3DecoderLayer,
    Ministral3MLP,
)

from transformers.models.mistral3.modeling_mistral3 import Mistral3Model
from transformers.models.pixtral.modeling_pixtral import PixtralAttention, PixtralAttentionLayer, PixtralMLP, PixtralVisionModel

from ..components import Attention, ImageScatter, Layer, Mlp, PixtralVision, VisionAttention, VisionLayer, VisionMlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    # The same text model inside a multimodal wrapper, loaded with task="image-text-to-text":
    # the mistral3 wrapper of Ministral 3 and Mistral 3.
    "model.language_model.embed_tokens": "embed_tokens",
    "model.language_model.layers": "layers",
    "model.language_model.norm": "norm",
    # Pixtral's tower (Mistral 3, Llava-Pixtral) and the projector. The tower's inner keys are relative to the
    # tower (multi-component, or names no text block has), so they bind on it alone.
    "model.vision_tower": "vision",
    "model.multi_modal_projector": "projector",
    "patch_conv": "patch_embed",
    "transformer.layers": "layers",
    "attention": "self_attn",
    "feed_forward": "mlp",
    "attention_norm": "input_layernorm",
    "ffn_norm": "post_attention_layernorm",
}


class Layer(Layer):
    """Ministral 3's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Ministral 3's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """Ministral 3's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {
    Ministral3DecoderLayer: Layer, Ministral3Attention: Attention, Ministral3MLP: Mlp,
    # Pixtral's pre-norm blocks on the shared attention interface, over the packed row of every image's patches.
    PixtralVisionModel: PixtralVision, PixtralAttentionLayer: VisionLayer, PixtralAttention: VisionAttention,
    PixtralMLP: VisionMlp,
    Mistral3Model: ImageScatter,  # the wrapper's forward scatters the image features: vision.image_features
}
