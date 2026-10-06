"""Qwen3-VL-MoE, text (``Qwen3VLMoeTextModel``, model_type ``qwen3_vl_moe_text``).

The language model inside ``Qwen3VLMoeForConditionalGeneration`` (model_type
``qwen3_vl_moe``): ``qwen3_vl_text``'s block, M-RoPE and DeepStack
(`Layer.deepstack_output` on the first blocks), with Qwen3-MoE's sparse
mixture of experts as the MLP (a softmax router, ``gate``, aliased ``router``;
dense blocks, ``Qwen3VLMoeTextMLP``, where ``mlp_only_layers`` or
``decoder_sparse_step`` say). Every checkpoint loads as the wrapper
(``task="image-text-to-text"``), whose text stack sits at ``model.language_model``.

The wrapper's Qwen ViT ``model.visual`` is ``vision`` (a `QwenVision`, as on
``qwen3_vl_text``); its ``merger`` is ``projector``.
"""

from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeModel, Qwen3VLMoeTextAttention, Qwen3VLMoeTextDecoderLayer, Qwen3VLMoeTextMLP, Qwen3VLMoeTextModel,
    Qwen3VLMoeTextSparseMoeBlock, Qwen3VLMoeVisionAttention, Qwen3VLMoeVisionBlock, Qwen3VLMoeVisionMLP,
    Qwen3VLMoeVisionModel,
)

from ..components import ImageScatter, Attention, Mlp, Moe, PackedVisionAttention, PackedVisionLayer, PackedVisionMlp, QwenVision
from . import qwen3_vl_text

#: The wrappers (config ``model_type``) whose tower's merged output is what they scatter into the text stream.
IMAGE_WRAPPERS = ("qwen3_vl_moe",)

RENAME = {**qwen3_vl_text.RENAME, "gate": "router"}


class Layer(qwen3_vl_text.Layer):
    """Qwen3-VL-MoE's decoder block: ``qwen3_vl_text``'s, ``deepstack_output`` on the first blocks."""


class TextModel(qwen3_vl_text.TextModel):
    """The text model whose forward adds the deepstack features: instrumented at build."""


class Attention(Attention):
    """Qwen3-VL-MoE's attention: M-RoPE applied after the q/k norms, before the shared interface, so the base holds."""


class Mlp(Mlp):
    """Qwen3-VL-MoE's dense MLP, on the blocks without a mixture; the residual is added in the block, so the base holds."""


class Moe(Moe, Mlp):
    """Qwen3-VL-MoE's mixture of experts: a softmax router and routed experts, no shared expert; the module returns the routed output, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {
    Qwen3VLMoeModel: ImageScatter,  # the wrapper's forward scatters the image features: vision.image_features
    Qwen3VLMoeTextModel: TextModel, Qwen3VLMoeTextDecoderLayer: Layer, Qwen3VLMoeTextAttention: Attention,
    Qwen3VLMoeTextMLP: Mlp, Qwen3VLMoeTextSparseMoeBlock: Moe,
    # The packed Qwen ViT: one attention call per image, so the interior is unavailable.
    Qwen3VLMoeVisionModel: QwenVision, Qwen3VLMoeVisionBlock: PackedVisionLayer,
    Qwen3VLMoeVisionAttention: PackedVisionAttention, Qwen3VLMoeVisionMLP: PackedVisionMlp,
}
