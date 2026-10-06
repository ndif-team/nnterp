"""HyperCLOVA X (``HyperCLOVAXForCausalLM``, NAVER's HyperCLOVA X SEED Think).

Llama's tree with a sandwich block and Granite's µP multipliers::

    h = x + post_norm1(self_attn(input_layernorm(x))) * residual_multiplier
    out = h + post_norm2(mlp(post_attention_layernorm(h))) * residual_multiplier

``post_norm1`` / ``post_norm2`` are RMS norms under ``use_post_norm`` and
identities otherwise; ``post_attention_layernorm`` is the pre-MLP norm (the Llama
meaning). What the block adds is each post-norm's output times the multiplier, so
``attention_output`` and ``mlp_output`` are that product: a computed copy, divided
by the multiplier on assignment, with a transform carrying an in-place edit back
into the post-norm's output, as on Granite; both modules keep the config, where
the multiplier is read. ``attention_multiplier`` is the
softmax scale passed to the shared interface; ``embedding_multiplier`` scales the
embedding module's output before the first block (``token_embeddings`` times it is
``layers[0].input``); ``logits_scaling`` *multiplies* the head's output (Granite
divides), so the family's ``project_on_vocab`` does the same.
"""

from typing import TYPE_CHECKING

import torch
from transformers.models.hyperclovax.modeling_hyperclovax import (
    HyperCLOVAXAttention,
    HyperCLOVAXDecoderLayer,
    HyperCLOVAXMLP,
)

from ..components import Attention, EProperty, Layer, Mlp, Residual
from .granitemoe import scaled_back

if TYPE_CHECKING:
    from ..standardized import StandardizedTransformer

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """HyperCLOVA X's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """HyperCLOVA X's attention: the shared eager forward; the block adds its post-norm's output times ``residual_multiplier``."""

    @EProperty("../post_norm1.output", description="What the attention adds to the residual stream: the post-attention norm's output times residual_multiplier")
    def attention_output(self, value) -> Residual:
        return value * self._module.config.residual_multiplier

    @attention_output.postprocess
    def attention_output(self, value):
        return value / self._module.config.residual_multiplier

    @attention_output.transform
    def attention_output(self, value, raw):
        return scaled_back(value, raw, self._module.config.residual_multiplier)


class Mlp(Mlp):
    """HyperCLOVA X's MLP; the block adds its post-norm's output times ``residual_multiplier``."""

    @EProperty("../post_norm2.output", description="What the MLP adds to the residual stream: the post-MLP norm's output times residual_multiplier")
    def mlp_output(self, value) -> Residual:
        return value * self._module.config.residual_multiplier

    @mlp_output.postprocess
    def mlp_output(self, value):
        return value / self._module.config.residual_multiplier

    @mlp_output.transform
    def mlp_output(self, value, raw):
        return scaled_back(value, raw, self._module.config.residual_multiplier)


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {HyperCLOVAXDecoderLayer: Layer, HyperCLOVAXAttention: Attention, HyperCLOVAXMLP: Mlp}


def project_on_vocab(model: "StandardizedTransformer", hidden: torch.Tensor) -> torch.Tensor:
    """The logit lens as the model makes its logits: the final norm, ``lm_head``, then times ``logits_scaling``."""
    return model.lm_head(model.norm(hidden)) * model.config.logits_scaling
