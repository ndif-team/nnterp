"""Mistral (``MistralForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward.

Two towers sit at ``model.vision_tower``, which is ``vision``, with the projector
``model.multi_modal_projector`` as ``projector``: Pixtral (a `PixtralVision`, every
image's patches packed in one row, no final norm) on the ``mistral3`` wrapper
(Mistral Small 3.1 / 3.2) and on Pixtral-12B's ``llava`` wrapper; CLIP on LLaVA-NeXT
(``llava-v1.6-mistral``) and BakLLaVA (``llava``). The towers' inner names differ, so
both sets are keyed. ``vision.image_features`` is read at the scatter
(`ImageScatter`), so on LLaVA-NeXT it is the unpadded projector output with its
newline tokens, as the text model receives it. Pixtral-12B reads
``vision_feature_layer`` -1 with strategy ``"full"``, so its ``projector.input`` is
``vision.tower_output``; BakLLaVA reads block -2 without the CLS token, as Llava 1.5.
"""

from transformers.models.clip.modeling_clip import CLIPAttention, CLIPEncoderLayer, CLIPMLP, CLIPVisionModel
from transformers.models.llava.modeling_llava import LlavaModel
from transformers.models.llava_next.modeling_llava_next import LlavaNextModel
from transformers.models.mistral.modeling_mistral import MistralAttention, MistralDecoderLayer, MistralMLP
from transformers.models.mistral3.modeling_mistral3 import Mistral3Model
from transformers.models.pixtral.modeling_pixtral import PixtralAttention, PixtralAttentionLayer, PixtralMLP, PixtralVisionModel

from ..components import Attention, ImageScatter, Layer, Mlp, PixtralVision, Vision, VisionAttention, VisionLayer, VisionMlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    # The same text model inside a multimodal wrapper, loaded with task="image-text-to-text":
    # the mistral3 wrapper of Mistral Small 3.1 and 3.2.
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
    # ... and CLIP's (LLaVA-NeXT, BakLLaVA).
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
    # Pixtral's pre-norm blocks on the shared attention interface, over the packed row of every image's patches.
    PixtralVisionModel: PixtralVision, PixtralAttentionLayer: VisionLayer, PixtralAttention: VisionAttention,
    PixtralMLP: VisionMlp,
    # CLIP's pre-norm blocks on the shared attention interface: the vision components hold as they are.
    CLIPVisionModel: Vision, CLIPEncoderLayer: VisionLayer, CLIPAttention: VisionAttention, CLIPMLP: VisionMlp,
    # The wrappers' models, whose forward scatters the image features: vision.image_features.
    Mistral3Model: ImageScatter, LlavaModel: ImageScatter, LlavaNextModel: ImageScatter,
}
