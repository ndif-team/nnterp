"""Cohere 2 (``Cohere2ForCausalLM``, Command-R7B / Command-A).

Cohere's tree and parallel block (``x + attn + mlp`` from one LayerNorm), with
sliding-window and full attention layers interleaved (``config.layer_types``).
Only the sliding layers apply the rotary; the full layers use no position
encoding. Both kinds make the same ``attention_interface`` call, so every
interior value resolves on every block, and on a full layer
``attention_queries`` / ``attention_keys`` are the unrotated projections.

The model multiplies the head's output by ``config.logit_scale``, as Cohere's
does, so the family takes Cohere's ``project_on_vocab``.

Aya Vision and Cohere2-Vision put a SigLIP tower at ``model.vision_tower``, which is
``vision`` (its ``post_layernorm`` over the patches is ``vision.norm``), and a
pixel-shuffling projector ``model.multi_modal_projector``, which is ``projector``;
``vision.image_features`` is read at the scatter (`ImageScatter`). Aya Vision's
projector reads the last block's ``layer_output`` before ``vision.norm``
(``vision_feature_layer`` -1, strategy ``"full"``) and norms after its pixel shuffle;
Cohere2-Vision's reads ``vision.tower_output`` and has no norm. Aya Vision 32B's text
config is ``cohere``, so it loads through that family, which binds no tower.
"""

from transformers.models.aya_vision.modeling_aya_vision import AyaVisionModel
from transformers.models.cohere2.modeling_cohere2 import Cohere2Attention, Cohere2DecoderLayer, Cohere2MLP
from transformers.models.cohere2_vision.modeling_cohere2_vision import Cohere2VisionModel
from transformers.models.siglip.modeling_siglip import SiglipAttention, SiglipEncoderLayer, SiglipMLP, SiglipVisionModel

from ..components import Attention, ImageScatter, Layer, Mlp, Vision, VisionAttention, VisionLayer, VisionMlp
from .cohere import project_on_vocab  # noqa: F401  the same head: the same logit_scale

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    # The same text model inside a multimodal wrapper, loaded with task="image-text-to-text":
    # AyaVisionForConditionalGeneration, Cohere2VisionForConditionalGeneration.
    "model.language_model.embed_tokens": "embed_tokens",
    "model.language_model.layers": "layers",
    "model.language_model.norm": "norm",
    # Aya Vision's and Cohere2-Vision's SigLIP tower and projector. The tower's inner keys are relative to the
    # tower (multi-component, or names no text block has), so they bind on it alone.
    "model.vision_tower": "vision",
    "model.multi_modal_projector": "projector",
    "embeddings.patch_embedding": "patch_embed",
    "encoder.layers": "layers",
    "post_layernorm": "norm",
    "layer_norm1": "input_layernorm",
    "layer_norm2": "post_attention_layernorm",
}


class Layer(Layer):
    """Cohere-2's parallel block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Cohere-2's attention; the shared eager forward on sliding and full layers alike, so the base holds."""


class Mlp(Mlp):
    """Cohere-2's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {
    Cohere2DecoderLayer: Layer, Cohere2Attention: Attention, Cohere2MLP: Mlp,
    # SigLIP's pre-norm blocks on the shared attention interface: the vision components hold as they are.
    SiglipVisionModel: Vision, SiglipEncoderLayer: VisionLayer, SiglipAttention: VisionAttention, SiglipMLP: VisionMlp,
    # The wrappers' models, whose forward scatters the image features: vision.image_features.
    AyaVisionModel: ImageScatter, Cohere2VisionModel: ImageScatter,
}
