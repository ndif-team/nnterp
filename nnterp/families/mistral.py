"""Mistral (``MistralForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward.

LLaVA-NeXT (``llava-v1.6-mistral``) and BakLLaVA (``llava``) put a CLIP tower at
``model.vision_tower``, which is ``vision``, with the projector
``model.multi_modal_projector`` as ``projector``. ``vision.image_features`` is read at the scatter
(`ImageScatter`), so on LLaVA-NeXT it is the unpadded projector output with its
newline tokens, as the text model receives it.
"""

from transformers.models.clip.modeling_clip import CLIPAttention, CLIPEncoderLayer, CLIPMLP, CLIPVisionModel
from transformers.models.llava.modeling_llava import LlavaModel
from transformers.models.llava_next.modeling_llava_next import LlavaNextModel
from transformers.models.mistral.modeling_mistral import MistralAttention, MistralDecoderLayer, MistralMLP

from ..components import Attention, ImageScatter, Layer, Mlp, Vision, VisionAttention, VisionLayer, VisionMlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    # The same text model inside a multimodal wrapper, loaded with task="image-text-to-text":
    # the mistral3 wrapper of Mistral Small 3.1 and 3.2.
    "model.language_model.embed_tokens": "embed_tokens",
    "model.language_model.layers": "layers",
    "model.language_model.norm": "norm",
    # LLaVA-NeXT's and BakLLaVA's CLIP tower and projector. The tower's inner keys are relative to the tower
    # (multi-component, or names no text block has), so they bind on it alone.
    "model.vision_tower": "vision",
    "model.multi_modal_projector": "projector",
    "embeddings.patch_embedding": "patch_embed",
    "encoder.layers": "layers",
    "layer_norm1": "input_layernorm",
    "layer_norm2": "post_attention_layernorm",
}


class Layer(Layer):
    """Mistral's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Mistral's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """Mistral's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {
    MistralDecoderLayer: Layer, MistralAttention: Attention, MistralMLP: Mlp,
    # CLIP's pre-norm blocks on the shared attention interface: the vision components hold as they are.
    CLIPVisionModel: Vision, CLIPEncoderLayer: VisionLayer, CLIPAttention: VisionAttention, CLIPMLP: VisionMlp,
    # The wrappers' models, whose forward scatters the image features: vision.image_features.
    LlavaModel: ImageScatter, LlavaNextModel: ImageScatter,
}
