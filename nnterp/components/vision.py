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


def scatter_host(model: Any) -> tuple[str, ImageScatter] | None:
    """The root's child whose forward scatters the image features (an `ImageScatter`), with its name, or ``None``."""
    return next(((name, child) for name, child in model._named_children() if isinstance(child, ImageScatter)), None)


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
    name, host = scatter_host(vision.root)
    return f"/{name}.source.{host.scatter}.inputs"


def scattered_argument(vision: Vision) -> int | str:
    """Which argument of the scatter is the image features: `ImageScatter.scatter_argument`."""
    return scatter_host(vision.root)[1].scatter_argument


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
    """

    sourced = True
    #: The operation of this module's forward that writes the image features into the token embeddings
    #: (Idefics 3 and SmolVLM: ``"self_inputs_merger_0"``, the helper call).
    scatter = "inputs_embeds_masked_scatter_0"
    #: Which of its arguments is the image features: a position, or a keyword's name (``"image_hidden_states"``).
    scatter_argument: int | str = 1


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
        name, host = scatter_host(self.root)
        args, kwargs = self.root.get(f"{name}.source.{host.scatter}").inputs
        argument = host.scatter_argument
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
