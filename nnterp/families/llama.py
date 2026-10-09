"""Llama (``LlamaForCausalLM``), the family the standard vocabulary is taken from.

Block names (``input_layernorm``, ``self_attn``, ``post_attention_layernorm``,
``mlp``) and ``lm_head`` are already the standard ones. The only change is
lifting the containers out of ``model.model``: ``model.layers`` instead of
``model.model.layers``. Multimodal wrappers around a Llama text model keep it at
``model.language_model`` (Llava, DeepSeek-VL, Janus) or ``model.text_model``
(Idefics 3, SmolVLM); ``RENAME`` carries those spellings too.

Llava 1.5 (``LlavaForConditionalGeneration``, ``model_type`` ``llava``), VipLlava
and LLaVA-NeXT add a CLIP tower at ``model.vision_tower``, which is ``vision`` (a
`Vision`), and the projector ``model.multi_modal_projector``, which is
``projector``. Llava feeds the projector ``vision.layers[-2].layer_output``
without its CLS token (``vision_feature_layer=-2``,
``vision_feature_select_strategy="default"``), not the tower's output; VipLlava
concatenates several blocks' streams; LLaVA-NeXT unpads the projector's output and
adds a newline token per row. ``vision.image_features`` is read at the scatter
(`ImageScatter`), so it is what enters the text model on all three. CLIP's
``post_layernorm`` norms only the pooled CLS token, never the patches, so it is
not ``vision.norm``.

DeepSeek-VL keeps a SigLIP tower at ``model.vision_model`` with the projector
``model.aligner``; Idefics 3 and SmolVLM keep their SigLIP-shaped ViT at
``model.vision_model`` (one row per image tile) with the pixel-shuffling
``model.connector`` as ``projector``, and merge the features in a helper,
``inputs_merger``. Those towers' ``post_layernorm`` does norm the patches; since a
rename key cannot tell it from CLIP's, `SiglipVision` serves it as ``vision.norm``.
"""

from transformers.models.clip.modeling_clip import CLIPAttention, CLIPEncoderLayer, CLIPMLP, CLIPVisionModel
from transformers.models.deepseek_vl.modeling_deepseek_vl import DeepseekVLModel
from transformers.models.idefics3.modeling_idefics3 import (
    Idefics3EncoderLayer, Idefics3Model, Idefics3VisionAttention, Idefics3VisionMLP, Idefics3VisionTransformer,
)
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer, LlamaMLP
from transformers.models.llava.modeling_llava import LlavaModel
from transformers.models.llava_next.modeling_llava_next import LlavaNextModel
from transformers.models.siglip.modeling_siglip import SiglipAttention, SiglipEncoderLayer, SiglipMLP, SiglipVisionModel
from transformers.models.smolvlm.modeling_smolvlm import (
    SmolVLMEncoderLayer, SmolVLMModel, SmolVLMVisionAttention, SmolVLMVisionMLP, SmolVLMVisionTransformer,
)
from transformers.models.vipllava.modeling_vipllava import VipLlavaModel

from ..components import Attention, ImageScatter, Layer, Mlp, Vision, VisionAttention, VisionLayer, VisionMlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    # The same text model inside a multimodal wrapper, loaded with task="image-text-to-text":
    # Llava 1.5, VipLlava, LLaVA-NeXT, DeepSeek-VL, Janus ...
    "model.language_model.embed_tokens": "embed_tokens",
    "model.language_model.layers": "layers",
    "model.language_model.norm": "norm",
    # ... and Idefics 3 / SmolVLM.
    "model.text_model.embed_tokens": "embed_tokens",
    "model.text_model.layers": "layers",
    "model.text_model.norm": "norm",
    # Llava's CLIP tower and projector; DeepSeek-VL's SigLIP and Idefics 3's ViT at model.vision_model.
    # The tower's inner keys are relative to the tower (multi-component, or names no text block has),
    # so they bind on it alone.
    "model.vision_tower": "vision",
    "model.vision_model": "vision",
    "model.multi_modal_projector": "projector",
    "model.aligner": "projector",
    "model.connector": "projector",
    "embeddings.patch_embedding": "patch_embed",
    "encoder.layers": "layers",
    "layer_norm1": "input_layernorm",
    "layer_norm2": "post_attention_layernorm",
}


class Layer(Layer):
    """Llama's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Llama's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """Llama's MLP; the residual is added in the block, so the base holds."""


class SiglipVision(Vision):
    """SigLIP's tower (DeepSeek-VL) and Idefics 3's and SmolVLM's: ``post_layernorm`` norms the patches, so it is ``norm``."""

    @property
    def norm(self):
        """The final norm over the patches, ``post_layernorm``: a property, since in this family CLIP's is not one."""
        return self.post_layernorm


class InputsMerger(ImageScatter):
    """Idefics 3's and SmolVLM's model: the features go in through ``inputs_merger(..., image_hidden_states=...)``."""

    scatter = "self_inputs_merger_0"
    scatter_argument = "image_hidden_states"


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {
    LlamaDecoderLayer: Layer, LlamaAttention: Attention, LlamaMLP: Mlp,
    # CLIP's pre-norm blocks on the shared attention interface: the vision components hold as they are.
    CLIPVisionModel: Vision, CLIPEncoderLayer: VisionLayer, CLIPAttention: VisionAttention, CLIPMLP: VisionMlp,
    SiglipVisionModel: SiglipVision, SiglipEncoderLayer: VisionLayer, SiglipAttention: VisionAttention, SiglipMLP: VisionMlp,
    Idefics3VisionTransformer: SiglipVision, Idefics3EncoderLayer: VisionLayer, Idefics3VisionAttention: VisionAttention,
    Idefics3VisionMLP: VisionMlp,
    SmolVLMVisionTransformer: SiglipVision, SmolVLMEncoderLayer: VisionLayer, SmolVLMVisionAttention: VisionAttention,
    SmolVLMVisionMLP: VisionMlp,
    # The wrappers' models, whose forward writes the image features in: vision.image_features.
    LlavaModel: ImageScatter, VipLlavaModel: ImageScatter, LlavaNextModel: ImageScatter, DeepseekVLModel: ImageScatter,
    Idefics3Model: InputsMerger, SmolVLMModel: InputsMerger,
}
