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
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from jaxtyping import Float
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
    not necessarily ``tower_output``.

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

    # -- availability ------------------------------------------------------------------

    def support(self, layer: int | None = None) -> dict[str, Any]:
        """Which tower values this checkpoint has, the way `StandardizedTransformer.support` reports the text model's.

        The tower's own values plus every block value over ``layers``
        (``"layer_output"``, ``"self_attn.attention_probabilities"``, ...):
        ``None`` when available on every block, else ``{layer: reason}``. With
        ``layer``, that block's values alone.
        """
        if layer is not None:
            return blocks_support(self.layers, layer)
        return {**super().support(), **blocks_support(self.layers)}
