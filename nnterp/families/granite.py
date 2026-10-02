"""Granite (``GraniteForCausalLM``, IBM Granite 3.x / 4.x dense).

Llama's tree and names, with four scalars from the config applied around it:

- ``embedding_multiplier``: the model multiplies the embedding module's output
  by it before the first block, so ``token_embeddings`` (the module's output)
  times the multiplier is ``layers[0].input``.
- ``attention_multiplier``: the attention's softmax scale in place of
  ``1/sqrt(head_dim)``, passed to the shared interface; nothing to override.
- ``residual_multiplier``: the block adds ``h * residual_multiplier`` for each
  sublayer, not ``h``. What reaches the residual stream is the scaled term, so
  ``attention_output`` and ``mlp_output`` are the module's output times the
  multiplier. That product is a computed copy, so each value divides on
  assignment, and a transform divides an edited copy back into the module's
  output so in-place edits reach the model.
- ``logits_scaling``: the model divides the head's output by it, so the family
  defines ``project_on_vocab`` with that step and the logit lens on the last
  block equals ``logits``.
"""

from typing import TYPE_CHECKING

import torch
from nnsight.intervention.envoy import Envoy
from transformers.models.granite.modeling_granite import GraniteAttention, GraniteDecoderLayer, GraniteMLP

from ..components import Attention, EProperty, Layer, Mlp, Residual, first_tensor, rewrap

if TYPE_CHECKING:
    from ..standardized import StandardizedTransformer

MODEL_TYPES = ("granite",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


def residual_multiplier(envoy: Envoy) -> float:
    """What the block multiplies each sublayer's output by before adding it."""
    return envoy._module.config.residual_multiplier


def scaled_back(envoy: Envoy, edited: torch.Tensor, raw) -> object:
    """The module output an edited scaled copy stands for: ``edited / residual_multiplier``, rewrapped.

    An unedited copy hands the module's own output back untouched, so a read
    leaves the forward bit-identical even where dividing by the multiplier
    would round.
    """
    output = first_tensor(raw)
    if torch.equal(edited, output * residual_multiplier(envoy)):
        return raw
    unscaled = edited / residual_multiplier(envoy)
    return (unscaled, *raw[1:]) if isinstance(raw, tuple) else unscaled


class Layer(Layer):
    """Granite's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Granite's attention; the shared eager forward, but the block adds its output times ``residual_multiplier``."""

    @EProperty(key="output", description="What the attention adds to the residual stream: its output times residual_multiplier")
    def attention_output(self, value) -> Residual:
        return first_tensor(value) * residual_multiplier(self)

    @attention_output.postprocess
    def attention_output(self, value):
        return rewrap(self, value / residual_multiplier(self))

    @attention_output.transform
    def attention_output(self, value, raw):
        return scaled_back(self, value, raw)


class Mlp(Mlp):
    """Granite's MLP; the block adds its output times ``residual_multiplier``."""

    @EProperty(key="output", description="What the MLP adds to the residual stream: its output times residual_multiplier")
    def mlp_output(self, value) -> Residual:
        return first_tensor(value) * residual_multiplier(self)

    @mlp_output.postprocess
    def mlp_output(self, value):
        return rewrap(self, value / residual_multiplier(self))

    @mlp_output.transform
    def mlp_output(self, value, raw):
        return scaled_back(self, value, raw)


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {GraniteDecoderLayer: Layer, GraniteAttention: Attention, GraniteMLP: Mlp}


def project_on_vocab(model: "StandardizedTransformer", hidden: torch.Tensor) -> torch.Tensor:
    """The logit lens as the model makes its logits: the final norm, ``lm_head``, then divided by ``logits_scaling``."""
    return model.lm_head(model.norm(hidden)) / model.config.logits_scaling
