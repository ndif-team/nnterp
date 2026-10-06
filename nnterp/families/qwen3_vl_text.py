"""Qwen3-VL, text (``Qwen3VLTextModel``, model_type ``qwen3_vl_text``).

The language model inside ``Qwen3VLForConditionalGeneration`` (model_type
``qwen3_vl``): Qwen3's block (``q_norm``/``k_norm`` inside the attention,
``head_dim`` the config's) under interleaved multimodal rotary embeddings
(M-RoPE), folded into one ``cos``/``sin`` pair before any block runs and
applied inside the attention after the q/k norms and before the shared
interface, so the base `Attention` holds. Every checkpoint loads as the
wrapper (``task="image-text-to-text"``), whose text stack sits at
``model.language_model``.

**DeepStack.** The tower taps three of its blocks
(``vision_config.deepstack_visual_indexes``), merges each tap with its own
``deepstack_merger_list[k]``, and the text model adds merger ``k``'s output at
the image positions after text block ``k``, outside the block, in its own
forward (``_deepstack_process``). So on blocks ``k < 3`` (fewer on a model
with fewer blocks) the stream entering block ``k + 1`` is not block ``k``'s
``layer_output`` at the image positions. `Layer.deepstack_output` serves what
is added there, ``[image_tokens, hidden]``:
``layers[k+1].input[mask] == layers[k].layer_output[mask] + layers[k].deepstack_output``,
the rest of the stream unchanged; on the other blocks it is `Unavailable` and
``layers[k+1].input == layers[k].layer_output``.

The wrapper's Qwen ViT ``model.visual`` is ``vision`` (a `QwenVision`: packed,
``[1, patches, vision_hidden]``, with a learned ``pos_embed`` added after
``patch_embed``); its ``merger`` is ``projector``. The deepstack mergers keep
their native path, ``vision.deepstack_merger_list[k]``.
"""

from typing import Any

from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel, Qwen3VLTextAttention, Qwen3VLTextDecoderLayer, Qwen3VLTextMLP, Qwen3VLTextModel,
    Qwen3VLVisionAttention, Qwen3VLVisionBlock, Qwen3VLVisionMLP, Qwen3VLVisionModel,
)

from ..components import (
    Attention, EProperty, ImageFeatures, ImageScatter, Layer, Mlp, QwenVision, QwenVisionAttention, Standard,
    VisionLayer, VisionMlp, pinned,
)

#: The text model's call that adds a deepstack feature after a block: ``_deepstack_process(hidden, mask, embeds)``.
DEEPSTACK = "self__deepstack_process_0"

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


def block_index(layer: Layer) -> int:
    """The block's index in ``model.layers``."""
    return int(layer.path.rsplit(".", 1)[-1])


def deepstack_blocks(layer: Layer) -> int:
    """How many blocks receive a deepstack feature: one per tapped tower block, at most the text model's depth."""
    config = layer.root.config
    vision = getattr(config, "vision_config", None)
    taps = len(getattr(vision, "deepstack_visual_indexes", None) or ())
    return min(taps, config.get_text_config().num_hidden_layers)


def no_deepstack(layer: Layer) -> str | None:
    """Why this block has no ``deepstack_output``, or ``None``."""
    model = layer.root
    if "vision" not in model._aliases:
        return "a text-only load: no tower, so no deepstack feature reaches the text model; load with task='image-text-to-text'"
    reason = model.vision.no_images()
    if reason:
        return reason
    count = deepstack_blocks(layer)
    if block_index(layer) >= count:
        return (
            f"the text model adds deepstack features after blocks 0..{count - 1} only "
            f"(one per vision_config.deepstack_visual_indexes); here layers[i+1].input == layers[i].layer_output"
        )
    return None


class DeepstackEProperty(EProperty):
    """An `EProperty` at the text model's ``_deepstack_process`` call, pinned to this block's occurrence of it.

    The call runs once after each of the first blocks, from one line of the
    text model's loop, so its location is the same for every block and block
    ``k``'s is its ``k``-th occurrence. A read or write here asks for that
    occurrence explicitly (`pinned`) rather than the next one the model reaches.
    """

    def __get__(self, obj: Any, owner: Any = None) -> Any:
        if obj is None:
            return self
        self._check(obj)
        with pinned(block_index(obj)):
            return super().__get__(obj, owner)

    def __set__(self, obj: Any, value: Any) -> None:
        self._check(obj)
        with pinned(block_index(obj)):
            super().__set__(obj, value)


class Layer(Layer):
    """Qwen3-VL's decoder block: returns a bare tensor, and the first blocks carry the deepstack feature added after them."""

    @DeepstackEProperty(
        f"../../source.{DEEPSTACK}.inputs",
        select=2,
        description="The deepstack image features the text model adds at the image tokens after this block",
        unavailable=no_deepstack,
    )
    def deepstack_output(self, value) -> ImageFeatures:
        """What the text model adds at the image tokens after this block, ``[image_tokens, hidden]``, flat in scatter order.

        Merger ``k`` of the tower's deepstack (``vision.deepstack_merger_list[k]``,
        fed by tower block ``vision_config.deepstack_visual_indexes[k]``) for block
        ``k``, so ``layers[k+1].input[image_token_mask] == layers[k].layer_output[image_token_mask] + deepstack_output``.
        Assign a tensor of the same shape to replace it, or edit it in place.
        Never reached on a text-only trace.
        """
        return value


class TextModel(Standard):
    """The text model whose forward adds the deepstack features: instrumented at build so the blocks can read its call."""

    sourced = True


class Attention(Attention):
    """Qwen3-VL's attention: M-RoPE is folded into ``cos``/``sin`` before the block and applied after the q/k norms, before the shared interface, so the base holds."""


class Mlp(Mlp):
    """Qwen3-VL's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {
    Qwen3VLModel: ImageScatter,  # the wrapper's forward scatters the image features: vision.image_features
    Qwen3VLTextModel: TextModel, Qwen3VLTextDecoderLayer: Layer, Qwen3VLTextAttention: Attention, Qwen3VLTextMLP: Mlp,
    # The Qwen ViT: one attention call per image, so its attention is a QwenVisionAttention.
    Qwen3VLVisionModel: QwenVision, Qwen3VLVisionBlock: VisionLayer,
    Qwen3VLVisionAttention: QwenVisionAttention, Qwen3VLVisionMLP: VisionMlp,
}
