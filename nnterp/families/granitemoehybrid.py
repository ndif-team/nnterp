"""GraniteMoE-Hybrid (``GraniteMoeHybridForCausalLM``, IBM Granite 4.0 and 4.0-H).

Llama's containers over one block class that holds either a Mamba-2 mixer or
attention, by ``config.layers_block_type``, and then a feed-forward::

    model.layers[i]          GraniteMoeHybridDecoderLayer
        .input_layernorm
        .mamba               GraniteMoeHybridMambaLayer, an SSD mixer   (or .self_attn)
        .post_attention_layernorm
        .block_sparse_moe    GraniteMoeHybridMoE (only when num_local_experts > 0)
        .shared_mlp          GraniteMoeHybridMLP (on every block)

    h = x + mixer(input_layernorm(x)) * residual_multiplier
    out = h + (block_sparse_moe(n) + shared_mlp(n)) * residual_multiplier     with n = post_attention_layernorm(h)

(``shared_mlp(n)`` alone on a dense checkpoint, Granite-4.0-H-Micro's.) ``mamba``
is ``linear_attn`` (`nnterp.StateSpace`) and ``shared_mlp``, the one feed-forward
every block has, is ``mlp``; each block has ``self_attn`` or ``linear_attn``,
never both. As on Granite (``granite.py``) the block adds each sublayer's output
times ``residual_multiplier``: both mixers' ``attention_output`` are the module's
output times it, computed copies divided on assignment and carried back by a
transform. No module returns the feed-forward's sum, so ``mlp_output`` is the
block's binding of it (``hidden_states_4`` with experts, ``hidden_states_5``
without) times the multiplier, and the family's `Layer` sets ``sourced`` for
it and hands the mixer and the shared expert, whose modules keep no config, the
multiplier (and the `Mlp` whether routed experts run) when it is built; a write lands on the sum, replacing both experts' contribution.
``mlp.output`` is the shared expert's output alone and
``layers[i].block_sparse_moe.output`` the routed experts'. The mixture's values
live on ``mlp`` (a `Moe`), which the block hands ``block_sparse_moe``'s ``router``
and ``experts`` envoys when it is built: ``shared_expert_output`` is ``mlp.output``
and ``routed_output + shared_expert_output == mlp_output / residual_multiplier``. ``mlp.intermediate_size``
and the root's ``intermediate_size`` are the shared expert's width,
``shared_intermediate_size``; the routed experts are ``intermediate_size`` wide.
The attention runs the shared eager forward with ``attention_multiplier`` as its
scale and no rotary under ``position_embedding_type`` ``nope`` (Granite-4.0-H);
``embedding_multiplier`` and ``logits_scaling`` are as on Granite.
"""

from typing import TYPE_CHECKING

from nnsight.intervention.envoy import Envoy
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
    GraniteMoeHybridAttention,
    GraniteMoeHybridDecoderLayer,
    GraniteMoeHybridMambaLayer,
    GraniteMoeHybridMLP,
)

from ..components import (
    EProperty, Layer, Moe, Residual, StateSpace, TokenEProperty, first_tensor, mixture_reason, rewrap,
)
from .granite import Attention as GraniteAttention
from .granite import project_on_vocab  # noqa: F401  the logit lens divides by logits_scaling, as Granite's
from .granitemoe import hand_residual_multiplier, scaled_back

if TYPE_CHECKING:
    from ..standardized import StandardizedTransformer

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "mamba": "linear_attn",
    "shared_mlp": "mlp",
}

#: The block's ``moe_hidden_states + self.shared_mlp(hidden_states)``: both experts' sum, before the multiplier.
EXPERTS_SUM = "hidden_states_4"
#: The block's ``self.shared_mlp(hidden_states)`` on a block without routed experts.
SHARED_ONLY = "hidden_states_5"


def _feed_forward_sum(envoy: Envoy) -> str:
    return f"../source.{EXPERTS_SUM if envoy.has_experts else SHARED_ONLY}.output"


class Layer(Layer):
    """GraniteMoE-Hybrid's block, either kind; returns a bare tensor.

    `Mlp.mlp_output` is a binding in this forward, read after the block has
    started (its mixer has returned), so the forward is instrumented at build.
    The Mamba-2 mixer's and the shared expert's modules keep no config, so the
    block hands them the multiplier, and its `Mlp` whether routed experts run
    beside the shared one and, when they do, the mixture's ``router`` and
    ``experts`` envoys, whose values the `Mlp` hosts.
    """

    sourced = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hand_residual_multiplier(self)
        self.mlp.has_experts = self._module.has_experts
        if self.mlp.has_experts:
            self.mlp.router = self.block_sparse_moe.router
            self.mlp.experts = self.block_sparse_moe.experts


class Attention(GraniteAttention):
    """GraniteMoE-Hybrid's attention: Granite's, the block adds its output times ``residual_multiplier``."""


class StateSpace(StateSpace):
    """GraniteMoE-Hybrid's Mamba-2 mixer: transformers' Mamba-2 scan; the block adds its output times ``residual_multiplier``."""

    #: Set by the block (`hand_residual_multiplier`): the mixer's module keeps no config.
    residual_multiplier: float

    @EProperty(key="output", description="What the Mamba-2 mixer adds to the residual stream: its output times residual_multiplier")
    def attention_output(self, value) -> Residual:
        return first_tensor(value) * self.residual_multiplier

    @attention_output.postprocess
    def attention_output(self, value):
        return rewrap(self, value / self.residual_multiplier)

    @attention_output.transform
    def attention_output(self, value, raw):
        return scaled_back(value, raw, self.residual_multiplier)


class Mlp(Moe):
    """GraniteMoE-Hybrid's shared expert, standing for the block's feed-forward: the block adds it plus the routed experts, times ``residual_multiplier``.

    It hosts the block's mixture of experts (``block_sparse_moe``, whose
    ``router`` and ``experts`` the block hands it), and is itself the shared
    expert.
    """

    SCORING = "topk_softmax"

    #: Set by the block: its multiplier, and whether routed experts run beside the shared one.
    residual_multiplier: float
    has_experts: bool

    def no_mixture(self) -> str | None:
        return None if self.has_experts else "this block runs no routed experts (num_local_experts is 0)"

    @TokenEProperty("output", description="The shared expert's output, unscaled", unavailable=mixture_reason)
    def shared_expert_output(self, value) -> Residual:
        """This MLP's own output: the shared expert beside the routed ones, before the block's multiplier."""
        return value

    @property
    def intermediate_size(self) -> int:
        """Width of the shared expert's hidden layer: the module's ``hidden_size`` (``shared_intermediate_size``)."""
        return self._module.hidden_size

    @EProperty(_feed_forward_sum, description="What the MLP adds to the residual stream: the routed experts plus the shared expert, times residual_multiplier")
    def mlp_output(self, value) -> Residual:
        return value * self.residual_multiplier

    @mlp_output.postprocess
    def mlp_output(self, value):
        return value / self.residual_multiplier

    @mlp_output.transform
    def mlp_output(self, value, raw):
        return scaled_back(value, raw, self.residual_multiplier)


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {
    GraniteMoeHybridDecoderLayer: Layer,
    GraniteMoeHybridAttention: Attention,
    GraniteMoeHybridMambaLayer: StateSpace,
    GraniteMoeHybridMLP: Mlp,
}


def intermediate_size(model: "StandardizedTransformer") -> int:
    """The MLP width is the shared expert's, ``shared_intermediate_size`` (``mlp`` is the shared expert); ``intermediate_size`` is the routed experts'."""
    return model.config.shared_intermediate_size
