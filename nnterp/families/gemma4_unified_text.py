"""Gemma 4 unified, text (``Gemma4UnifiedForCausalLM``, model_type ``gemma4_unified_text``): Gemma-4-12B.

Gemma-4's tree and block without per-layer embeddings or a mixture of experts:
the sandwich, then ``out = (x + post_attention_layernorm(...) +
post_feedforward_layernorm(...)) * layer_scalar``, in place. The contributions are the
post-norms' outputs, served unscaled, so the identity is
``(input + attention_output + mlp_output) * layer_scalar == layer_output``. KV
sharing, ``attention_k_eq_v`` and the per-layer ``head_dim`` are Gemma-4's
(`gemma4_text`), and so is the attention envoy (keys and values served as
copies private to the block), and so are the root's ``head_dim`` and
``num_kv_heads``: the config's top-level values. A ``gemma4_unified`` checkpoint
(``Gemma4UnifiedForConditionalGeneration``) keeps the text stack at
``model.language_model``, so ``RENAME`` carries both spellings.

The wrapper is encoder-free: no vision tower, no blocks. ``model.embed_vision``
embeds the processor's merged patches (``model_patch_size`` pixels square)
directly: ``patch_ln1``, ``patch_dense``, ``patch_ln2``, a factorized 2D position
embedding, ``pos_norm``, then ``multimodal_embedder`` (an RMS norm and a linear
onto the text width, the same projection Gemma-4's ``embed_vision`` is). So
``vision`` is the embedder (a `Vision` with no ``layers``), ``vision.patch_embed``
its ``patch_dense``, and ``projector`` its ``multimodal_embedder``, the module
mapping onto the text width as on every other wrapper. The processor pads each
image's patches to ``max_soft_tokens`` rows at position ``(-1, -1)``, and the
embedder runs on every row; the wrapper strips the padded rows of the
projector's output before scattering, so ``vision.image_features`` is read at
the scatter (the first ``inputs_embeds.masked_scatter`` of
``Gemma4UnifiedModel.forward``, an `ImageScatter`).
"""

from typing import Any

from transformers.models.gemma4_unified.modeling_gemma4_unified import (
    Gemma4UnifiedModel, Gemma4UnifiedTextAttention, Gemma4UnifiedTextDecoderLayer, Gemma4UnifiedTextMLP,
    Gemma4UnifiedVisionEmbedder,
)

from ..components import EProperty, ImageScatter, Layer, Mlp, Patches, Residual, Standard, Unavailable, Vision
from ..components.vision import variable_resolution
from . import gemma4_text
from .gemma4_text import head_dim, num_kv_heads  # noqa: F401  the sizes read the same config keys

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    # A Gemma4UnifiedForConditionalGeneration: the same text model under ``model.language_model``.
    "model.language_model.embed_tokens": "embed_tokens",
    "model.language_model.layers": "layers",
    "model.language_model.norm": "norm",
    # The wrapper's encoder-free embedder and its projection onto the text width.
    "model.embed_vision": "vision",
    "model.embed_vision.multimodal_embedder": "projector",
    "patch_dense": "patch_embed",
}


class Layer(Layer):
    """Gemma-4 unified's decoder block; returns a bare tensor (the sum times ``layer_scalar``), so the base holds."""


class Attention(gemma4_text.Attention):
    """Gemma-4 unified's attention: Gemma-4's, with its post-norm contribution and its keys and values private to the block."""


class Mlp(Mlp):
    """Gemma-4 unified's MLP: what reaches the residual stream is the post-feedforward norm's output."""

    @EProperty(
        "../post_feedforward_layernorm.output",
        description="What the MLP adds to the residual stream: the post-feedforward norm's output",
    )
    def mlp_output(self, value) -> Residual:
        return value


def _no_blocks(vision: Vision, name: str) -> Unavailable:
    return Unavailable(f"{vision.path}.{name} is not available: the wrapper is encoder-free, its image embedder has no attention blocks")


class Vision(Vision):
    """The encoder-free image embedder: a `Vision` with no blocks.

    ``patch_embeddings`` is ``patch_dense``'s output and ``tower_output`` the
    embedder's states before the projection (``projector.input``), both over
    the padded patches, ``[images, max_soft_tokens, mm_embed_dim]``.
    ``image_features`` is read at the wrapper's scatter, after the padded rows
    are stripped. ``num_layers`` is 0; ``hidden_size`` is ``mm_embed_dim`` and
    ``patch_size`` the merged patch the embedder sees (``model_patch_size``).
    """

    @property
    def num_layers(self) -> int:
        return 0

    @property
    def hidden_size(self) -> int:
        return self._module.patch_dense.out_features

    @property
    def patch_size(self) -> int:
        """Side of one merged patch, in pixels: what one image token embeds."""
        return self.root.config.vision_config.model_patch_size

    @property
    def num_heads(self) -> int:
        raise _no_blocks(self, "num_heads")

    @property
    def head_dim(self) -> int:
        raise _no_blocks(self, "head_dim")

    @property
    def intermediate_size(self) -> int:
        raise _no_blocks(self, "intermediate_size")

    image_size = property(variable_resolution)

    @EProperty("multimodal_embedder.input", description="The embedder's states over the patches before the projection")
    def tower_output(self, value) -> Patches:
        """The embedder's states before the projection, ``[images, max_soft_tokens, mm_embed_dim]``, padded rows included.

        What the projector receives (``projector.input``). Assign to replace it.
        """
        return value

    def support(self, layer: int | None = None) -> dict[str, Any]:
        """The embedder's values (no block values: it has no blocks); empty on a load no image reaches."""
        if layer is not None:
            raise IndexError(f"{self.path} has no blocks")
        return {} if self.no_images() else Standard.support(self)


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {
    Gemma4UnifiedTextDecoderLayer: Layer, Gemma4UnifiedTextAttention: Attention, Gemma4UnifiedTextMLP: Mlp,
    Gemma4UnifiedVisionEmbedder: Vision,
    Gemma4UnifiedModel: ImageScatter,  # the wrapper's forward scatters the image features: vision.image_features
}
