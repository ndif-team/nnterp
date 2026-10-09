"""`Vision`: an image-text-to-text checkpoint's vision tower, and its blocks.

A multimodal wrapper is a text model plus a vision tower, a projector, and a
step that scatters the projected features into the text stream at the image
tokens. The text model keeps the standard names it has everywhere; the tower
answers to ``model.vision``, its blocks to ``vision.layers[i]``, and the last
module before the scatter to ``model.projector``. The family names them in its
``RENAME`` (keys that resolve only on the wrapper's tree) and keys these envoys
on the tower's module types in its ``ENVOYS``.

The tower's blocks are pre-norm attention + MLP blocks on transformers' shared
attention interface (SigLIP, CLIP), so they are `Layer`, `Attention` and `Mlp`
with the stream values re-annotated as `Patches`: ``layer_input``, ``layer_output``,
``attention_output`` and ``mlp_output`` mean what they mean on a text block,
``vision.layers[i].layer_input + attention_output + mlp_output == layer_output``. The
attention interior is inherited unchanged: ``attention_probabilities`` is a
`Pattern` whose ``batch`` axis is the tower's (one row per image) and whose
``query``/``key`` axes are the image's patches. There is no causal mask (a
patch attends to every patch of its image); a tower that masks does so for
its own layout (Pixtral's block-diagonal mask between packed images, Gemma
4's padded keys). The tower's sizes are on `Vision`, read off its own config;
the root's sizes stay the text model's.

Where the image meets the text model is the tower's too: ``vision.image_token_mask``
(which positions of the text batch hold an image token, off the root's inputs)
and ``vision.image_features`` (what the wrapper scatters into the token
embeddings, flat over those tokens, read at the scatter in the forward of the
module the family keys `ImageScatter` on), so
``layers[0].input[vision.image_token_mask] == vision.image_features``.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from .attention import Attention, HeadOutputs, Keys, Queries, Values
from .eproperty import EProperty, Unavailable
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


def variable_resolution(vision: Vision) -> int:
    """``image_size`` on a tower with no fixed resolution: `Unavailable`, saying where each image's size is."""
    raise Unavailable(
        f"{vision.path}.image_size is not available: the tower takes images of any resolution, each cut into its own "
        "patch grid by the processor; read the grid off the processor's output (image_grid_thw, image_sizes, "
        "image_position_ids)"
    )


def no_image_tokens(vision: Vision) -> str | None:
    """Why ``vision.image_token_mask`` is unavailable, or ``None``."""
    reason = vision.no_images()
    if reason is None and image_token_id(vision.root.config) is None:
        reason = "the config names no image_token_id"
    return reason


def scatter_host(model: Any) -> tuple[str, Any] | None:
    """The module whose forward scatters the image features, with its path from the root, or ``None``.

    The root's child the family keys `ImageScatter` on, or the root itself
    (path ``""``) on a wrapper that scatters in its own forward, whose family
    names the operation in ``ROOT_SCATTER`` (Llama 4).
    """
    if getattr(model.family, "ROOT_SCATTER", None) and "projector" in model._aliases:
        return "", model
    return next(((name, child) for name, child in model._named_children() if isinstance(child, ImageScatter)), None)


def scatter_call(model: Any) -> tuple[str, int | str]:
    """The scatter's path from the root (``"model.source.inputs_embeds_masked_scatter_0"``) and which argument is the image features."""
    name, host = scatter_host(model)
    if not name:
        return f"source.{model.family.ROOT_SCATTER}", 1
    return f"{name}.source.{host.scatter}", host.scatter_argument


def no_image_features(vision: Vision) -> str | None:
    """Why ``vision.image_features`` is unavailable, or ``None``.

    ``image_features`` is read at the scatter, in the forward of the module the
    family keys `ImageScatter` on; a wrapper whose family keys none there binds
    the tower's names but does not serve the value.
    """
    reason = no_image_tokens(vision)
    if reason is None and scatter_host(vision.root) is None:
        model = vision.root
        reason = (
            f"the {model.family.__name__.rsplit('.', 1)[-1]} family keys no ImageScatter on the "
            f"{model.config.model_type!r} wrapper, so where its image features enter the text stream is unknown"
        )
    return reason


def image_scatter(vision: Vision) -> str:
    """The path of the scatter's arguments, from the root: ``"/model.source.inputs_embeds_masked_scatter_0.inputs"``."""
    return f"/{scatter_call(vision.root)[0]}.inputs"


def scattered_argument(vision: Vision) -> int | str:
    """Which argument of the scatter is the image features: `ImageScatter.scatter_argument`."""
    return scatter_call(vision.root)[1]


def no_tower_run(envoy: Any) -> str | None:
    """Why the tower ``envoy`` belongs to never runs (`Vision.no_images`), or ``None``: the gate on every tower value."""
    while envoy is not None and not isinstance(envoy, Vision):
        envoy = envoy.parent
    return envoy.no_images() if envoy is not None else None


def as_patches(value: torch.Tensor) -> torch.Tensor:
    """A tower's stream as `Patches`: a packed tower's ``[patches, vision_hidden]`` gets a leading images axis of 1 (a view)."""
    return value.unsqueeze(0) if value.dim() == 2 else value


def as_native(current: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """`Patches` back in the shape of the module's own tensor ``current``: a packed tower's drops the leading 1."""
    return value.squeeze(0) if current.dim() == 2 and value.dim() == 3 else value


class ImageScatter(Standard):
    """The wrapper's inner model (``model.model``): the module whose forward scatters the image features into the token embeddings.

    A family keys it on the wrapper model's type (``LlavaModel: ImageScatter``);
    `Vision.image_features` reads the `scatter_argument` of the `scatter`
    operation, the features in ``inputs_embeds.masked_scatter(image_mask,
    image_features)``, which is what the text model receives at the image
    tokens whatever the wrapper did after its projector (LLaVA-NeXT's
    unpadding and newline tokens). A wrapper that writes the features in
    elsewhere subclasses it with its own operation and argument. The forward
    is instrumented at build (``sourced``), so the scatter is served after the
    tower's values, which run inside the same forward. Carries no values of
    its own.

    A wrapper that scatters in the root's own forward (Llama 4's
    ``Llama4ForConditionalGeneration``) has no child to key: its family sets
    ``ROOT_SCATTER`` to the operation's name, the root is the host
    (`scatter_host`), and `StandardizedTransformer` instruments its own
    forward at build.
    """

    sourced = True
    #: The operation of this module's forward that writes the image features into the token embeddings
    #: (Idefics 3 and SmolVLM: ``"self_inputs_merger_0"``, the helper call).
    scatter = "inputs_embeds_masked_scatter_0"
    #: Which of its arguments is the image features: a position, or a keyword's name (``"image_hidden_states"``).
    scatter_argument: int | str = 1


class VisionLayer(Layer):
    """A vision tower's block: ``layer_input`` and ``layer_output`` are the tower's stream, `Patches`."""

    @EProperty(key="input", description="The tower's stream entering the block", unavailable=no_tower_run)
    def layer_input(self, value: torch.Tensor) -> Patches:
        """The tower's stream entering this block, ``[images, patches, vision_hidden]``, before any of its norms.

        The block's first argument, with a packed tower's leading 1 (as
        ``layer_output``); a view: assign or edit in place.
        """
        return as_patches(value)

    @layer_input.postprocess
    def layer_input(self, value: torch.Tensor) -> torch.Tensor:
        return as_native(self.input, value)

    @EProperty(key="output", description="The tower's stream leaving the block", unavailable=no_tower_run)
    def layer_output(self, value: Any) -> Patches:
        """The tower's stream leaving this block, ``[images, patches, vision_hidden]``: a view; assign or edit in place."""
        return as_patches(first_tensor(value))

    @layer_output.postprocess
    def layer_output(self, value: torch.Tensor) -> Any:
        return rewrap(self, as_native(first_tensor(self.output), value))


class VisionAttention(Attention):
    """A vision tower block's attention: what it adds to the tower's stream is `Patches`; the interior is `Attention`'s."""

    def off_interface(self) -> str | None:
        return no_tower_run(self) or super().off_interface()

    @EProperty(key="output", description="What the attention adds to the tower's stream", unavailable=no_tower_run)
    def attention_output(self, value: Any) -> Patches:
        """The attention sublayer's contribution to the tower's stream, ``[images, patches, vision_hidden]``."""
        return as_patches(first_tensor(value))

    @attention_output.postprocess
    def attention_output(self, value: torch.Tensor) -> Any:
        return rewrap(self, as_native(first_tensor(self.output), value))


class VisionMlp(Mlp):
    """A vision tower block's MLP: what it adds to the tower's stream is `Patches`."""

    @EProperty(key="output", description="What the MLP adds to the tower's stream", unavailable=no_tower_run)
    def mlp_output(self, value: Any) -> Patches:
        """The MLP sublayer's contribution to the tower's stream, ``[images, patches, vision_hidden]``."""
        return as_patches(first_tensor(value))

    @mlp_output.postprocess
    def mlp_output(self, value: torch.Tensor) -> Any:
        return rewrap(self, as_native(first_tensor(self.output), value))


class Vision(Standard):
    """``model.vision``: a vision tower's root. Its blocks are ``layers``; its sizes are its own config's.

    Values: ``patch_embeddings``, the patch embedding's output one row per patch
    (before any position embedding, CLS token or pre-norm the tower adds), and
    ``tower_output``, the last block's stream after the tower's final norm
    where it has one, before any pooling, CLS dropping or adapter (the tower's
    ``last_hidden_state``, unless the tower returns something after those).
    The host reads the tower its own way: Gemma 3 pools ``tower_output``,
    Llava 1.5 takes ``vision.layers[-2].layer_output`` and drops the CLS
    token, so what reaches the projector is ``projector.input``, not
    necessarily ``tower_output``. Where the image meets the text model:
    ``image_token_mask``, read off the model's inputs, and ``image_features``,
    read at the scatter (`ImageScatter`, keyed from the root), so
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
        """Side of the square image the tower is configured for, in pixels; `Unavailable` on a tower with no fixed resolution (`variable_resolution`)."""
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

    @EProperty("patch_embed.output", description="The patch embedding's output, one row per patch", unavailable=no_tower_run)
    def patch_embeddings(self, value: torch.Tensor) -> Patches:
        """The patch embedding's output, ``[images, patches, vision_hidden]``, patches in raster order.

        On a convolution, which returns ``[images, vision_hidden, rows,
        columns]``, a view of it with the grid flattened, so in-place edits
        land; on a linear patch embedding (Llama 4, Gemma 4), its output.
        Assign a tensor of the same shape to replace it. Position embeddings,
        a CLS token and a pre-norm come after.
        """
        return value.flatten(2).transpose(1, 2) if value.dim() == 4 else as_patches(value)

    @patch_embeddings.postprocess
    def patch_embeddings(self, value: torch.Tensor) -> torch.Tensor:
        current = self.patch_embed.output
        return value.transpose(1, 2).reshape(current.shape) if current.dim() == 4 else as_native(current, value)

    @EProperty(key="output", description="The last block's stream after the tower's final norm, before any pooling or adapter", unavailable=no_tower_run)
    def tower_output(self, value: Any) -> Patches:
        """The last block's stream after the final norm where the tower has one (`norm`), ``[images, patches, vision_hidden]``.

        Before any pooling, CLS dropping or adapter: the tower's
        ``last_hidden_state``, which a tower that returns something after
        those reads where the stream is instead. Assigning replaces it;
        whether that reaches the text model depends on what the host reads
        (see the class docstring).
        """
        return as_patches(value.last_hidden_state if hasattr(value, "last_hidden_state") else first_tensor(value))

    @tower_output.postprocess
    def tower_output(self, value: torch.Tensor) -> Any:
        output = self.output
        if hasattr(output, "last_hidden_state"):
            output.last_hidden_state = as_native(output.last_hidden_state, value)
            return output
        return rewrap(self, as_native(first_tensor(output), value))

    @EProperty(image_scatter, select=scattered_argument, description="The image features the text model receives at the image tokens, flat over them", unavailable=no_image_features)
    def image_features(self, value: torch.Tensor) -> ImageFeatures:
        """The image features the text model receives, ``[image_tokens, hidden]``, flat over every image token of the batch.

        Read at the scatter: the tensor the wrapper's forward writes into the
        token embeddings at the image tokens in row-major order (`ImageScatter`),
        so ``layers[0].input[image_token_mask] == image_features``. A view of
        it: in-place edits land, and an assigned tensor of the same shape
        replaces it. Read after the tower's values and before
        ``layers[0].input``. Never reached on a text-only trace.
        """
        return value.reshape(-1, value.shape[-1])

    @image_features.postprocess
    def image_features(self, value: torch.Tensor) -> torch.Tensor:
        path, argument = scatter_call(self.root)
        args, kwargs = self.root.get(path).inputs
        return value.reshape((kwargs[argument] if isinstance(argument, str) else args[argument]).shape)

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


# -- the Qwen ViT ------------------------------------------------------------------------

#: Why the Qwen ViT's pattern is unavailable: its attention runs per image.
PER_IMAGE = (
    "the Qwen ViT's attention makes one interface call per image (attention_interface_2; one per window on "
    "Qwen2.5-VL's windowed blocks), so no one tensor is the block's pattern; read attention_queries and "
    "attention_keys, whole, and split them where the calls do: at the attention's cu_seqlens argument "
    "(self_attn.inputs[1]['cu_seqlens']), which per image is the processor's image_grid_thw"
)


def no_concatenated_heads(envoy: Any) -> str | None:
    """Why the Qwen ViT's ``attention_head_outputs`` is unavailable, or ``None``: flash attention takes the other branch."""
    if "flash" in envoy._module.config._attn_implementation:
        return (
            "flash attention runs every image in one call over cu_seqlens, with no concatenation to read; "
            "load with attn_implementation='eager' or 'sdpa'"
        )
    return no_tower_run(envoy)


class QwenVisionAttention(VisionAttention):
    """The Qwen ViT's attention: one interface call per image (or window), so the interior is read whole around them.

    ``attention_queries``, ``attention_keys`` and ``attention_values`` are the
    whole ``[1, heads, patches, head_dim]`` tensors (the queries and keys after
    the 2D rotary embedding) before the module splits them per image, and
    ``attention_head_outputs`` is the per-image outputs concatenated back,
    ``[1, patches, heads, head_dim]``, under any implementation but flash.
    ``attention_scores`` and ``attention_probabilities`` are `Unavailable`
    (`PER_IMAGE`): loading eager does not help, since the module splits
    under every implementation.
    """

    def off_interface(self) -> str | None:
        return no_tower_run(self) or PER_IMAGE

    @EProperty("source.unsqueeze_0.output", description=Attention.attention_queries.description, unavailable=no_tower_run)
    def attention_queries(self, value: torch.Tensor) -> Queries:
        """Every image's queries, ``[1, heads, patches, head_dim]``, before the per-image split; assign or edit in place."""
        return value

    @EProperty("source.unsqueeze_1.output", description=Attention.attention_keys.description, unavailable=no_tower_run)
    def attention_keys(self, value: torch.Tensor) -> Keys:
        """Every image's keys, ``[1, heads, patches, head_dim]``, before the per-image split; assign or edit in place."""
        return value

    @EProperty("source.unsqueeze_2.output", description=Attention.attention_values.description, unavailable=no_tower_run)
    def attention_values(self, value: torch.Tensor) -> Values:
        """Every image's values, ``[1, heads, patches, head_dim]``, before the per-image split; assign or edit in place."""
        return value

    @EProperty("source.torch_cat_0.output", description=Attention.attention_head_outputs.description, unavailable=no_concatenated_heads)
    def attention_head_outputs(self, value: torch.Tensor) -> HeadOutputs:
        """Every image's per-head outputs concatenated back, ``[1, patches, heads, head_dim]``, before the output projection."""
        return value


class QwenVision(Vision):
    """The Qwen ViT (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3.5): packed, its merger inside, at ``model.visual``.

    The tower runs on ``[patches, vision_hidden]``, every image's patches
    concatenated, so its stream values are `Patches` with 1 in the images
    axis, ``[1, patches, vision_hidden]``, and its blocks' attention is a
    `QwenVisionAttention`.

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

    The tower has no final norm, so ``tower_output`` is the last block's
    stream, the merger's input. It is served at the tower's output, after the
    merger has run, so ``projector.input`` and ``projector.output`` are read
    before ``tower_output``.

    Sizes, off the vision config: ``hidden_size`` is the tower's width
    (``embed_dim`` on Qwen2-VL, whose config's ``hidden_size`` is the merger's
    output width), ``num_heads``, ``intermediate_size`` (Qwen2-VL's is
    ``embed_dim * mlp_ratio``), ``patch_size``, ``spatial_merge_size`` and
    ``window_size`` (Qwen2.5-VL's, in pixels; ``None`` on the others).
    ``image_size`` is `Unavailable`: the tower takes any resolution.
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

    image_size = property(variable_resolution)

    @property
    def spatial_merge_size(self) -> int:
        """Side of the square block of patches the merger folds into one image token."""
        return self._module.config.spatial_merge_size

    @property
    def window_size(self) -> int | None:
        """Side of an attention window in pixels (Qwen2.5-VL), or ``None`` on a tower without windows."""
        return getattr(self._module.config, "window_size", None)


class PixtralVision(Vision):
    """Pixtral's tower (Mistral 3, Llava-Pixtral): *packed*, every image's patches in one row.

    The convolution runs on the batch padded to its largest image, each image
    is cropped to its own grid, and the grids are flattened and concatenated
    into ``[1, all patches, vision_hidden]`` before ``ln_pre``; the blocks
    attend under a block-diagonal mask, so no patch sees another image's. So
    `Patches` has a leading 1 here, and ``patch_embeddings`` is that packed
    row as it enters ``ln_pre`` (the convolution's own output, ``patch_embed.output``,
    is still the padded grid). The processor's ``image_sizes`` split the row
    per image: ``(height // patch_size) * (width // patch_size)`` patches each,
    in order. The attention interior is whole: one interface call over the
    packed row, so ``attention_probabilities`` is ``[1, heads, all patches,
    all patches]``, zero between images. ``image_size`` is `Unavailable`: the
    config's is the largest side the processor resizes to, not one size every
    image has.
    """

    image_size = property(variable_resolution)

    @EProperty("ln_pre.input", description="Every image's patch embeddings packed in one row, entering the pre-norm", unavailable=no_tower_run)
    def patch_embeddings(self, value: torch.Tensor) -> Patches:
        """Every image's patch embeddings, ``[1, all patches, vision_hidden]``, image after image, each in raster order.

        What enters ``ln_pre``: the convolution's output cropped per image and
        concatenated. Assign a tensor of the same shape to replace it.
        """
        return value
