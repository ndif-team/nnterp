"""Llama (``LlamaForCausalLM``), the family the standard vocabulary is taken from.

Block names (``input_layernorm``, ``self_attn``, ``post_attention_layernorm``,
``mlp``) and ``lm_head`` are already the standard ones. The only change is
lifting the containers out of ``model.model``: ``model.layers`` instead of
``model.model.layers``. Multimodal wrappers around a Llama text model keep it at
``model.language_model`` (Llava, DeepSeek-VL, Janus) or ``model.text_model``
(Idefics 3, SmolVLM); ``RENAME`` carries those spellings too.

Llava 1.5 (``LlavaForConditionalGeneration``, ``model_type`` ``llava``) adds a CLIP
tower at ``model.vision_tower``, which is ``vision`` (a `Vision`), and the
projector ``model.multi_modal_projector``, which is ``projector``. Llava feeds the
projector ``vision.layers[-2].layer_output`` without its CLS token
(``vision_feature_layer=-2``, ``vision_feature_select_strategy="default"``), not
the tower's output, and the projector's output is what it scatters, so
``vision.image_features`` is the projector's output. CLIP's ``post_layernorm`` norms only
the pooled CLS token, never the patches, so it is not ``vision.norm``.
"""

from transformers.models.clip.modeling_clip import CLIPAttention, CLIPEncoderLayer, CLIPMLP, CLIPVisionModel
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer, LlamaMLP

from ..components import Attention, Layer, Mlp, Vision, VisionAttention, VisionLayer, VisionMlp

#: The wrappers (config ``model_type``) whose projector's output is what they scatter into the text stream.
#: LLaVA-NeXT binds the same names but unpads and adds newline tokens after its projector, so it is not here.
IMAGE_WRAPPERS = ("llava",)

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
    # Llava's CLIP tower and projector. The tower's inner keys are relative to the tower
    # (multi-component, or names no text block has), so they bind on it alone.
    "model.vision_tower": "vision",
    "model.multi_modal_projector": "projector",
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


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {
    LlamaDecoderLayer: Layer, LlamaAttention: Attention, LlamaMLP: Mlp,
    # CLIP's pre-norm blocks on the shared attention interface: the vision components hold as they are.
    CLIPVisionModel: Vision, CLIPEncoderLayer: VisionLayer, CLIPAttention: VisionAttention, CLIPMLP: VisionMlp,
}
