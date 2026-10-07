"""Qwen2 / Qwen2.5 (``Qwen2ForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward.

llava-interleave (a ``llava`` wrapper) and LLaVA-OneVision put a SigLIP tower at
``model.vision_tower``, which is ``vision`` (its ``post_layernorm`` over the patches
is ``vision.norm``), and the projector ``model.multi_modal_projector``, which is
``projector``. OneVision feeds the projector the image's crops and then unpads
its output and adds newline tokens, so ``vision.image_features``, read at the
scatter (`ImageScatter`), is not the projector's output there.
"""

from transformers.models.llava.modeling_llava import LlavaModel
from transformers.models.llava_onevision.modeling_llava_onevision import LlavaOnevisionModel
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2DecoderLayer, Qwen2MLP
from transformers.models.siglip.modeling_siglip import SiglipAttention, SiglipEncoderLayer, SiglipMLP, SiglipVisionModel

from ..components import Attention, ImageScatter, Layer, Mlp, Vision, VisionAttention, VisionLayer, VisionMlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    # The same text model inside a multimodal wrapper, loaded with task="image-text-to-text":
    # LLaVA-OneVision, llava-interleave, InternVL, FastVLM, GOT-OCR2.
    "model.language_model.embed_tokens": "embed_tokens",
    "model.language_model.layers": "layers",
    "model.language_model.norm": "norm",
    # llava-interleave's and LLaVA-OneVision's SigLIP tower and projector. The tower's inner keys are relative to
    # the tower (multi-component, or names no text block has), so they bind on it alone.
    "model.vision_tower": "vision",
    "model.multi_modal_projector": "projector",
    "embeddings.patch_embedding": "patch_embed",
    "encoder.layers": "layers",
    "post_layernorm": "norm",
    "layer_norm1": "input_layernorm",
    "layer_norm2": "post_attention_layernorm",
}


class Layer(Layer):
    """Qwen2's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Qwen2's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """Qwen2's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {
    Qwen2DecoderLayer: Layer, Qwen2Attention: Attention, Qwen2MLP: Mlp,
    # SigLIP's pre-norm blocks on the shared attention interface: the vision components hold as they are.
    SiglipVisionModel: Vision, SiglipEncoderLayer: VisionLayer, SiglipAttention: VisionAttention, SiglipMLP: VisionMlp,
    # The wrappers' models, whose forward scatters the image features: vision.image_features.
    LlavaModel: ImageScatter, LlavaOnevisionModel: ImageScatter,
}
