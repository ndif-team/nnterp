"""`Vision`: an image-text-to-text checkpoint's vision tower, and its blocks.

A multimodal wrapper is a text model plus a vision tower, a projector, and a
step that scatters the projected features into the text stream at the image
tokens. The text model keeps the standard names it has everywhere; the tower
answers to ``model.vision``, its blocks to ``vision.layers[i]``, and the module
whose output is scattered to ``model.projector``. The family names them in its
``RENAME`` (keys that resolve only on the wrapper's tree) and keys these envoys
on the tower's module types in its ``ENVOYS``.

The tower's blocks are pre-norm attention + MLP blocks on transformers' shared
attention interface (SigLIP, CLIP), so they are `Layer`, `Attention` and `Mlp`
with the stream values re-annotated as `Patches`: ``layer_output``,
``attention_output`` and ``mlp_output`` mean what they mean on a text block,
``vision.layers[i].input + attention_output + mlp_output == layer_output``. The
attention interior is inherited unchanged: ``attention_probabilities`` is a
`Pattern` whose ``batch`` axis is the tower's (one row per image) and whose
``query``/``key`` axes are the image's patches, unmasked (the tower attends
both ways). The tower's sizes are on `Vision`, read off its own config; the
root's sizes stay the text model's.

Where the image meets the text model is the tower's too: ``vision.image_token_mask``
(which positions of the text batch hold an image token, off the root's inputs)
and ``vision.image_features`` (the projector's output, flat over those tokens),
so ``layers[0].input[vision.image_token_mask] == vision.image_features``.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from .attention import Attention
from .eproperty import EProperty
from .layer import Layer
from .mlp import Mlp
from .standard import Standard, blocks_support, first_tensor, rewrap

#: A vision tower's stream: the tower's own batch (one row per image or crop) by its tokens (the patches,
#: with CLIP's CLS token first), ``vision_hidden`` wide. The tower blocks' ``layer_output``,
#: ``attention_output``, ``mlp_output``, and the tower's ``patch_embeddings`` and ``tower_output``.
Patches = Float[Tensor, "images patches vision_hidden"]
#: Which positions of the text batch hold an image token: ``vision.image_token_mask``.
ImageTokenMask = Bool[Tensor, "batch seq"]
#: What the text model receives at the image tokens, flat over every image token of the batch in row-major
#: (scatter) order, the text model's ``hidden`` wide: ``vision.image_features``.
ImageFeatures = Float[Tensor, "image_tokens hidden"]


def image_token_id(config: Any) -> int | None:
    """The id of the token the processor puts where an image's features go: ``image_token_id`` (``image_token_index`` on older configs)."""
    for name in ("image_token_id", "image_token_index"):
        value = getattr(config, name, None)
        if isinstance(value, int):
            return value
    return None


def no_image_tokens(vision: Vision) -> str | None:
    """Why ``vision.image_token_mask`` is unavailable, or ``None``."""
    reason = vision.no_images()
    if reason is None and image_token_id(vision.root.config) is None:
        reason = "the config names no image_token_id"
    return reason


def no_image_features(vision: Vision) -> str | None:
    """Why ``vision.image_features`` is unavailable, or ``None``.

    ``image_features`` is the projector's output, which is what the wrapper
    scatters into the text stream on the wrappers the family lists in
    ``IMAGE_WRAPPERS`` (each verified by the suite:
    ``layers[0].input[image_token_mask] == image_features``). A wrapper that
    rearranges the projector's output before the scatter (LLaVA-NeXT's
    unpadding and newline tokens) binds the same names but is not listed, so
    the value says so rather than serving the wrong tensor.
    """
    reason = no_image_tokens(vision)
    if reason is None:
        model = vision.root
        wrappers = tuple(getattr(model.family, "IMAGE_WRAPPERS", ()))
        if model.config.model_type not in wrappers:
            reason = (
                f"the {model.config.model_type!r} wrapper is not one whose projector output is known to be what it "
                f"scatters into the text stream (the {model.family.__name__.rsplit('.', 1)[-1]} family lists {wrappers})"
            )
    return reason


class VisionLayer(Layer):
    """A vision tower's block: ``layer_output`` is the tower's stream, `Patches`."""

    @EProperty(key="output", description="The tower's stream leaving the block")
    def layer_output(self, value: Any) -> Patches:
        """The tower's stream leaving this block, ``[images, patches, vision_hidden]``; assign or edit in place."""
        return first_tensor(value)

    @layer_output.postprocess
    def layer_output(self, value: torch.Tensor) -> Any:
        return rewrap(self, value)


class VisionAttention(Attention):
    """A vision tower block's attention: what it adds to the tower's stream is `Patches`; the interior is `Attention`'s."""

    @EProperty(key="output", description="What the attention adds to the tower's stream")
    def attention_output(self, value: Any) -> Patches:
        """The attention sublayer's contribution to the tower's stream, ``[images, patches, vision_hidden]``."""
        return first_tensor(value)

    @attention_output.postprocess
    def attention_output(self, value: torch.Tensor) -> Any:
        return rewrap(self, value)


class VisionMlp(Mlp):
    """A vision tower block's MLP: what it adds to the tower's stream is `Patches`."""

    @EProperty(key="output", description="What the MLP adds to the tower's stream")
    def mlp_output(self, value: Any) -> Patches:
        """The MLP sublayer's contribution to the tower's stream, ``[images, patches, vision_hidden]``."""
        return first_tensor(value)

    @mlp_output.postprocess
    def mlp_output(self, value: torch.Tensor) -> Any:
        return rewrap(self, value)


class Vision(Standard):
    """``model.vision``: a vision tower's root. Its blocks are ``layers``; its sizes are its own config's.

    Values: ``patch_embeddings``, the patch embedding's output one row per patch
    (before any position embedding, CLS token or pre-norm the tower adds), and
    ``tower_output``, what the tower returns over the patches (its
    ``last_hidden_state``). The host reads the tower its own way: Gemma 3 pools
    ``tower_output``, Llava 1.5 takes ``vision.layers[-2].layer_output`` and
    drops the CLS token, so what reaches the projector is ``projector.input``,
    not necessarily ``tower_output``. Where the image meets the text model:
    ``image_token_mask``, read off the model's inputs, and ``image_features``,
    the projector's output (keyed ``"/projector.output"``, from the root), so
    ``layers[0].input[image_token_mask] == image_features``. Both need an image
    to reach the model (`no_images`).

    The sizes (`num_layers`, `hidden_size`, `num_heads`, `head_dim`,
    `intermediate_size`, `patch_size`, `image_size`) are read off the tower's
    own config, the plain spellings SigLIP's and CLIP's use; a tower that
    spells one its own way overrides the property on a subclass.

    Attributes:
        layers: The tower's blocks, each a `VisionLayer`.
        patch_embed: The patch embedding (a convolution on SigLIP and CLIP).
        norm: The final norm over the patches, where the tower has one (SigLIP's ``post_layernorm``;
            CLIP's norms only the pooled CLS token and is not aliased).
    """

    layers: Sequence[VisionLayer]

    # -- sizes (off the tower's own config) -----------------------------------------

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def hidden_size(self) -> int:
        return self._module.config.hidden_size

    @property
    def num_heads(self) -> int:
        return self._module.config.num_attention_heads

    @property
    def head_dim(self) -> int:
        """Width of one head: the config's ``head_dim`` when it says, else ``hidden_size // num_heads``."""
        return getattr(self._module.config, "head_dim", None) or self.hidden_size // self.num_heads

    @property
    def intermediate_size(self) -> int:
        return self._module.config.intermediate_size

    @property
    def patch_size(self) -> int:
        """Side of one square patch, in pixels."""
        return self._module.config.patch_size

    @property
    def image_size(self) -> int:
        """Side of the square image the tower is configured for, in pixels."""
        return self._module.config.image_size

    # -- values ----------------------------------------------------------------------

    @EProperty("/inputs", description="Which positions hold image tokens: input_ids == the config's image_token_id; read-only", unavailable=no_image_tokens)
    def image_token_mask(self, value: Any) -> ImageTokenMask:
        """Which positions of the text batch hold an image token, ``[batch, seq]`` bool: ``input_ids == config.image_token_id``.

        Read off the model's inputs, so like ``model.input_ids`` it is read
        before anything else in the invoke. All false on a text-only trace.
        Read-only.
        """
        return value[1]["input_ids"] == image_token_id(self.root.config)

    @image_token_mask.postprocess
    def image_token_mask(self, value: Any) -> Any:
        raise AttributeError("image_token_mask is derived from the ids and cannot be assigned; assign input_ids")

    @EProperty("patch_embed.output", description="The patch embedding's output, one row per patch")
    def patch_embeddings(self, value: torch.Tensor) -> Patches:
        """The patch embedding's output, ``[images, patches, vision_hidden]``, patches in raster order.

        A convolution returns ``[images, vision_hidden, rows, columns]``; this
        is a view of it with the grid flattened, so in-place edits land.
        Assign a tensor of the same shape to replace it. Position embeddings,
        CLIP's CLS token and its pre-norm come after.
        """
        return value.flatten(2).transpose(1, 2) if value.dim() == 4 else value

    @patch_embeddings.postprocess
    def patch_embeddings(self, value: torch.Tensor) -> torch.Tensor:
        current = self.patch_embed.output
        return value.transpose(1, 2).reshape(current.shape) if current.dim() == 4 else value

    @EProperty(key="output", description="What the tower returns over the patches: its last_hidden_state")
    def tower_output(self, value: Any) -> Patches:
        """The tower's output over the patches, ``[images, patches, vision_hidden]``: its ``last_hidden_state``.

        The last block's stream after the final norm, where the tower has one
        (`norm`). Assigning replaces it in the tower's output; whether that
        reaches the text model depends on what the host reads (see the class
        docstring).
        """
        return value.last_hidden_state if hasattr(value, "last_hidden_state") else first_tensor(value)

    @tower_output.postprocess
    def tower_output(self, value: torch.Tensor) -> Any:
        output = self.output
        if hasattr(output, "last_hidden_state"):
            output.last_hidden_state = value
            return output
        return rewrap(self, value)

    @EProperty("/projector.output", description="The image features the text model receives at the image tokens, flat over them", unavailable=no_image_features)
    def image_features(self, value: torch.Tensor) -> ImageFeatures:
        """The image features the text model receives, ``[image_tokens, hidden]``, flat over every image token of the batch.

        The projector's output, which the wrapper scatters into the token
        embeddings at the image tokens in row-major order, so
        ``layers[0].input[image_token_mask] == image_features``. A view of the
        projector's output: in-place edits land, and an assigned tensor of the
        same shape replaces it. Read after the tower's values and before
        ``layers[0].input``. Never reached on a text-only trace.
        """
        return value.reshape(-1, value.shape[-1])

    @image_features.postprocess
    def image_features(self, value: torch.Tensor) -> torch.Tensor:
        return value.reshape(self.root.projector.output.shape)

    # -- availability ------------------------------------------------------------------

    def no_images(self) -> str | None:
        """Why no image reaches the model this tower belongs to, or ``None``: the one place that decides.

        A wrapper whose family names no ``projector`` (where the image enters
        the text model is unknown), or a load without a processor (a
        ``task="text-generation"`` load of the wrapper).
        """
        model = self.root
        if getattr(model, "family", None) is None:  # a StandardizedTransformer's root has its family
            return "the tower is not part of a StandardizedTransformer, so the model's inputs and projector are unknown"
        if "projector" not in model._aliases:
            return f"the {model.family.__name__.rsplit('.', 1)[-1]} family names no projector on this wrapper"
        if getattr(model, "processor", None) is None:
            return "a text-only load: no processor, so no image reaches the model; load with task='image-text-to-text'"
        return None

    def support(self, layer: int | None = None) -> dict[str, Any]:
        """Which tower values this checkpoint has, the way `StandardizedTransformer.support` reports the text model's.

        The tower's own values (``image_token_mask`` and ``image_features``
        among them) plus every block value over ``layers`` (``"layer_output"``,
        ``"self_attn.attention_probabilities"``, ...): ``None`` when available
        on every block, else ``{layer: reason}``. With ``layer``, that block's
        values alone. Empty on a load no image reaches (`no_images`): the tower
        never runs, so the model's `support` has no ``vision`` rows, and a
        read of an image value raises `Unavailable` saying why.
        """
        if layer is not None:
            return blocks_support(self.layers, layer)
        if self.no_images():
            return {}
        return {**super().support(), **blocks_support(self.layers)}


# -- packed towers ------------------------------------------------------------------------
# A packed tower (the Qwen ViT) runs on ``[patches, vision_hidden]``: every image's patches
# concatenated, one attention call per image (or window) over its own run of rows. Its
# stream values are served with a leading images axis of 1, as `Patches`; the attention
# interior is unavailable, since no one tensor is the block's pattern.

#: Why the attention interior is unavailable on a packed tower.
PACKED = (
    "a packed tower: the attention makes one interface call per image (attention_interface_2; one per window "
    "on Qwen2.5-VL's windowed blocks), so the block's pattern is not one tensor; attention_output and the block "
    "values are whole"
)


def unpack(value: torch.Tensor) -> torch.Tensor:
    """A packed tower's ``[patches, vision_hidden]`` as `Patches`, ``[1, patches, vision_hidden]``: a view."""
    return value.unsqueeze(0) if value.dim() == 2 else value


def pack(value: torch.Tensor) -> torch.Tensor:
    """`Patches` ``[1, patches, vision_hidden]`` back to the packed tower's ``[patches, vision_hidden]``."""
    return value.squeeze(0) if value.dim() == 3 else value


class PackedVisionLayer(VisionLayer):
    """A packed tower's block: ``layer_output`` is ``[1, patches, vision_hidden]``, every image's patches in one row."""

    @EProperty(key="output", description="The tower's stream leaving the block, every image's patches in one row")
    def layer_output(self, value: Any) -> Patches:
        """The tower's stream leaving this block, ``[1, patches, vision_hidden]``: a view; assign or edit in place."""
        return unpack(first_tensor(value))

    @layer_output.postprocess
    def layer_output(self, value: torch.Tensor) -> Any:
        return rewrap(self, pack(value))


class PackedVisionAttention(VisionAttention):
    """A packed tower block's attention: ``attention_output`` is whole; the interior is unavailable (`PACKED`).

    The module calls the attention interface once per image, inside a list
    comprehension, whatever ``attn_implementation`` (flash takes one call with
    ``cu_seqlens`` instead), so a value read at a call is one image's and
    loading eager does not help: `off_interface` says so, ahead of `needs_eager`.
    """

    def off_interface(self) -> str | None:
        return PACKED

    @EProperty(key="output", description="What the attention adds to the tower's stream")
    def attention_output(self, value: Any) -> Patches:
        """The attention sublayer's contribution to the tower's stream, ``[1, patches, vision_hidden]``."""
        return unpack(first_tensor(value))

    @attention_output.postprocess
    def attention_output(self, value: torch.Tensor) -> Any:
        return rewrap(self, pack(value))


class PackedVisionMlp(VisionMlp):
    """A packed tower block's MLP: ``mlp_output`` is ``[1, patches, vision_hidden]``."""

    @EProperty(key="output", description="What the MLP adds to the tower's stream")
    def mlp_output(self, value: Any) -> Patches:
        """The MLP sublayer's contribution to the tower's stream, ``[1, patches, vision_hidden]``."""
        return unpack(first_tensor(value))

    @mlp_output.postprocess
    def mlp_output(self, value: torch.Tensor) -> Any:
        return rewrap(self, pack(value))


class PackedVision(Vision):
    """A packed tower's root: ``patch_embeddings`` and ``tower_output`` are ``[1, patches, vision_hidden]``."""

    @EProperty("patch_embed.output", description="The patch embedding's output, one row per patch")
    def patch_embeddings(self, value: torch.Tensor) -> Patches:
        """The patch embedding's output, ``[1, patches, vision_hidden]``, every image's patches in one row: a view."""
        return unpack(value)

    @patch_embeddings.postprocess
    def patch_embeddings(self, value: torch.Tensor) -> torch.Tensor:
        return pack(value)

    @EProperty(key="output", description="What the tower returns over the patches: its last_hidden_state")
    def tower_output(self, value: Any) -> Patches:
        """The last block's stream as the tower returns it (``last_hidden_state``), ``[1, patches, vision_hidden]``."""
        return unpack(value.last_hidden_state if hasattr(value, "last_hidden_state") else first_tensor(value))

    @tower_output.postprocess
    def tower_output(self, value: torch.Tensor) -> Any:
        output = self.output
        if hasattr(output, "last_hidden_state"):
            output.last_hidden_state = pack(value)
            return output
        return rewrap(self, pack(value))


class QwenVision(PackedVision):
    """The Qwen ViT (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3.5): packed, its merger inside, at ``model.visual``.

    The processor cuts each image into ``patch_size`` squares (two frames
    deep), and lays them out in ``spatial_merge_size`` x ``spatial_merge_size``
    blocks, each block's patches consecutive, images one after another; the
    merger folds each block into one image token. So ``patches`` runs over
    every image of the invoke, ``image_grid_thw`` (``[t, h, w]`` per image, in
    patches) splits it, and an image's patches are in merge-block order, not
    raster order. Qwen2.5-VL's tower further permutes them into attention
    windows at entry: its blocks' values, ``tower_output`` and the merger's
    output are in window order, and the tower restores the merge-block order
    after the merger.

    ``image_features`` is the tower's ``pooler_output``, the merged output in
    the order the wrapper scatters it (on Qwen2.5-VL, after the restore the
    merger's output has not had), so it is read at the tower's output rather
    than at the projector's.

    Sizes, off the vision config: ``hidden_size`` is the tower's width
    (``embed_dim`` on Qwen2-VL, whose config's ``hidden_size`` is the merger's
    output width), ``num_heads``, ``intermediate_size`` (Qwen2-VL's is
    ``embed_dim * mlp_ratio``), ``patch_size``, ``spatial_merge_size`` and
    ``window_size`` (Qwen2.5-VL's, in pixels; ``None`` on the others). There
    is no fixed ``image_size``: the tower takes any resolution.
    """

    @property
    def hidden_size(self) -> int:
        config = self._module.config
        return getattr(config, "embed_dim", None) or config.hidden_size

    @property
    def num_heads(self) -> int:
        return self._module.config.num_heads

    @property
    def intermediate_size(self) -> int:
        config = self._module.config
        size = getattr(config, "intermediate_size", None)
        return size if size else int(self.hidden_size * config.mlp_ratio)

    @property
    def image_size(self) -> None:
        """``None``: the tower takes any resolution (``image_grid_thw`` says what each image was cut into)."""
        return None

    @property
    def spatial_merge_size(self) -> int:
        """Side of the square block of patches the merger folds into one image token."""
        return self._module.config.spatial_merge_size

    @property
    def window_size(self) -> int | None:
        """Side of an attention window in pixels (Qwen2.5-VL), or ``None`` on a tower without windows."""
        return getattr(self._module.config, "window_size", None)

    @EProperty("output", description="The image features the text model receives at the image tokens, flat over them", unavailable=no_image_features)
    def image_features(self, value: Any) -> ImageFeatures:
        """The image features the text model receives, ``[image_tokens, hidden]``: the tower's ``pooler_output``.

        The merger's output in scatter order, so
        ``layers[0].input[image_token_mask] == image_features``; the same tensor
        as ``projector.output`` except on Qwen2.5-VL, whose merger output is in
        window order. In-place edits land, and an assigned tensor of the same
        shape replaces it. Never reached on a text-only trace.
        """
        return value.pooler_output

    @image_features.postprocess
    def image_features(self, value: torch.Tensor) -> Any:
        output = self.output
        output.pooler_output = value
        return output
