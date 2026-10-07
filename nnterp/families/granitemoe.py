"""GraniteMoE (``GraniteMoeForCausalLM``, IBM Granite 3.x MoE).

Granite's block and multipliers (``granite.py``) with a mixture of experts for
the MLP: ``block_sparse_moe`` (``GraniteMoeMoE``, routed experts only) is
``mlp``. The block adds ``h * residual_multiplier`` for each sublayer, so
``attention_output`` and ``mlp_output`` are the module's output times the
multiplier, computed copies that divide on assignment and carry an in-place edit
back through a transform, as on Granite. The mixture keeps no config, so its
`Mlp` reads the multiplier off its parent block (`residual_multiplier`). ``embedding_multiplier`` scales the
embedding module's output before the first block and ``logits_scaling`` divides
the head's output (the family's ``project_on_vocab``, Granite's). Every block has
the mixture, so the config's ``intermediate_size`` is the experts' width. The
mixture is a `Moe`: its router returns ``(indices, weights, logits)``, and
``router_logits`` is read where the router computes them. ``routed_output`` is the
module's output, unscaled: ``mlp_output == routed_output * residual_multiplier``.
"""

import torch
from nnsight.intervention.envoy import Envoy
from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeAttention, GraniteMoeDecoderLayer, GraniteMoeMoE

from ..components import EProperty, Layer, Moe, Residual, first_tensor, rewrap
from .granite import Attention as GraniteAttention
from .granite import project_on_vocab  # noqa: F401  the logit lens divides by logits_scaling, as Granite's

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "block_sparse_moe": "mlp",
}


def residual_multiplier(envoy: Envoy) -> float:
    """The block's ``residual_multiplier``, off the parent block's module: a mixture's or a mixer's module keeps no config."""
    return envoy.parent._module.residual_multiplier


def scaled_back(edited: torch.Tensor, raw, multiplier: float):
    """The value an edited scaled copy stands for: ``edited / multiplier``, rewrapped; an unedited copy hands ``raw`` back untouched."""
    served = first_tensor(raw)
    if torch.equal(edited, served * multiplier):
        return raw
    unscaled = edited / multiplier
    return (unscaled, *raw[1:]) if isinstance(raw, tuple) else unscaled


class Layer(Layer):
    """GraniteMoE's decoder block; returns a bare tensor, so the base holds."""


class Attention(GraniteAttention):
    """GraniteMoE's attention: Granite's, the block adds its output times ``residual_multiplier``."""


class Mlp(Moe):
    """GraniteMoE's mixture of experts; the block adds its output times ``residual_multiplier``, read off the block."""

    SCORING = "topk_softmax"

    residual_multiplier = property(residual_multiplier)

    @EProperty(key="output", description="What the MLP adds to the residual stream: its output times residual_multiplier")
    def mlp_output(self, value) -> Residual:
        return first_tensor(value) * self.residual_multiplier

    @mlp_output.postprocess
    def mlp_output(self, value):
        return rewrap(self, value / self.residual_multiplier)

    @mlp_output.transform
    def mlp_output(self, value, raw):
        return scaled_back(value, raw, self.residual_multiplier)


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {GraniteMoeDecoderLayer: Layer, GraniteMoeAttention: Attention, GraniteMoeMoE: Mlp}
