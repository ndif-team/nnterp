"""GraniteMoE with a shared expert (``GraniteMoeSharedForCausalLM``).

GraniteMoE's block (``granitemoe.py``) with a dense shared expert beside the
mixture: the block computes ``block_sparse_moe(h) + shared_mlp(h)`` from the
pre-MLP norm's output ``h`` and adds that sum times ``residual_multiplier``. No
module returns the sum, so ``block_sparse_moe`` is ``mlp`` (it is on every block;
``shared_mlp`` is ``None`` when ``shared_intermediate_size`` is 0) and its
``mlp_output`` is the block's binding of the sum (``hidden_states_4``, or
``hidden_states_3``, the mixture's output alone, without a shared expert) times
the multiplier: what the block adds. As on Granite it is a computed copy, divided
on assignment and carried back by a transform after an in-place edit; the write
lands on the sum, so it replaces both experts' contribution. ``mlp.output`` is the
mixture's output alone and ``layers[i].shared_mlp.output`` the shared expert's.
The binding is in the block's forward, read after the block has started, so the
family's `Layer` sets ``sourced``; the mixture keeps no config, so the `Layer` also
hands its `Mlp` the multiplier and whether the block has a shared expert
(``residual_multiplier``, ``shared``) when it is built. The attention, ``embedding_multiplier`` and
``logits_scaling`` are as on Granite. The mixture is a `Moe` (GraniteMoE's routing);
its ``shared_expert_output`` is the block's ``shared_mlp``'s output, unscaled, so
``routed_output + shared_expert_output == mlp_output / residual_multiplier``.
"""

from nnsight.intervention.envoy import Envoy
from transformers.models.granitemoeshared.modeling_granitemoeshared import (
    GraniteMoeSharedAttention,
    GraniteMoeSharedDecoderLayer,
    GraniteMoeSharedMoE,
)

from ..components import EProperty, Layer, Moe, Residual
from .granite import Attention as GraniteAttention
from .granite import project_on_vocab  # noqa: F401  the logit lens divides by logits_scaling, as Granite's
from .granitemoe import hand_residual_multiplier, scaled_back

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "block_sparse_moe": "mlp",
}

#: The block's ``moe_hidden_states + self.shared_mlp(hidden_states)``: both experts' sum, before the multiplier.
EXPERTS_SUM = "hidden_states_4"
#: The block's ``hidden_states = moe_hidden_states`` when it has no shared expert.
EXPERTS_ONLY = "hidden_states_3"


def _experts_sum(envoy: Envoy) -> str:
    return f"../source.{EXPERTS_SUM if envoy.shared else EXPERTS_ONLY}.output"


def _no_shared_mlp(envoy: Envoy) -> str | None:
    return None if envoy.shared else "this block has no shared expert (shared_intermediate_size is 0)"


class Layer(Layer):
    """GraniteMoE-Shared's decoder block; returns a bare tensor.

    `Mlp.mlp_output` is a binding in this forward, read after the block has
    started (its attention has returned), so the forward is instrumented at build.
    The mixture's module keeps no config, so the block hands its `Mlp` the
    multiplier and whether a shared expert runs beside it.
    """

    sourced = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hand_residual_multiplier(self)
        self.mlp.shared = self._module.shared_mlp is not None


class Attention(GraniteAttention):
    """GraniteMoE-Shared's attention: Granite's, the block adds its output times ``residual_multiplier``."""


class Mlp(Moe):
    """GraniteMoE-Shared's mixture of experts; the block adds it plus the shared expert, times ``residual_multiplier``."""

    SCORING = "topk_softmax"

    #: Set by the block: its multiplier, and whether a shared expert runs beside the mixture.
    residual_multiplier: float
    shared: bool

    @EProperty(_experts_sum, description="What the MLP adds to the residual stream: the mixture plus the shared expert, times residual_multiplier")
    def mlp_output(self, value) -> Residual:
        return value * self.residual_multiplier

    @mlp_output.postprocess
    def mlp_output(self, value):
        return value / self.residual_multiplier

    @mlp_output.transform
    def mlp_output(self, value, raw):
        return scaled_back(value, raw, self.residual_multiplier)

    @EProperty("../shared_mlp.output", description=Moe.shared_expert_output.description, unavailable=_no_shared_mlp)
    def shared_expert_output(self, value) -> Residual:
        """The block's ``shared_mlp``'s output: the shared expert is the block's child, beside the mixture."""
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {GraniteMoeSharedDecoderLayer: Layer, GraniteMoeSharedAttention: Attention, GraniteMoeSharedMoE: Mlp}
