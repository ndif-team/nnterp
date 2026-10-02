"""Laguna (``LagunaForCausalLM``, poolside's Laguna).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, which returns a bare tensor. What differs is inside the
sublayers:

* **Per-block head counts.** ``num_attention_heads_per_layer`` gives each block
  its own number of query heads (48 on the full-attention blocks and 64 or 72 on
  the sliding-window ones in the released checkpoints); the key/value heads and
  ``head_dim`` are the same everywhere. ``layers[i].self_attn.num_heads`` is that
  block's count, read off the module; the root's ``num_heads`` is the config's
  top-level ``num_attention_heads``.
* **Gated attention.** q/k norms before the rotary (partial on the full blocks),
  the shared eager forward, then a softplus gate (``g_proj``, per head or per
  channel) on the interface's output before ``o_proj``: ``attention_head_outputs``
  are ungated and ``attention_output`` is the gated projection the block adds.
* **Mixture of experts.** ``mlp_layer_types`` names the dense blocks
  (``LagunaMLP``, ``intermediate_size`` wide) and the sparse ones
  (``LagunaSparseMoeBlock``: routed experts ``moe_intermediate_size`` wide, times
  ``moe_routed_scaling_factor``, plus a shared expert, as one tensor). The shared
  expert (``mlp.shared_experts``) is a ``LagunaMLP`` too, so it is an `Mlp`; its
  ``mlp_output`` is unavailable, since the block adds the mixture's sum.
"""

from typing import TYPE_CHECKING

from transformers.models.laguna.modeling_laguna import LagunaAttention, LagunaDecoderLayer, LagunaMLP, LagunaSparseMoeBlock

from ..components import Attention, EProperty, Layer, Mlp, Moe, Residual, TokenEProperty, first_tensor, rewrap

if TYPE_CHECKING:
    from nnsight.intervention.envoy import Envoy

MODEL_TYPES = ("laguna",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "gate": "router",
}


def _not_a_block_mlp(envoy: "Envoy") -> str | None:
    if envoy.path.rsplit(".", 1)[-1] == "mlp":
        return None
    return "this is the shared expert inside a mixture of experts; what the block adds is the mixture's output, at layers[i].mlp"


class Layer(Layer):
    """Laguna's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Laguna's attention; the shared eager forward, a gate after it, and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """Laguna's dense MLP or mixture of experts; both return what the block adds, so the base holds but on the shared expert."""

    @EProperty(key="output", description=Mlp.mlp_output.description, unavailable=_not_a_block_mlp)
    def mlp_output(self, value) -> Residual:
        return first_tensor(value)

    @mlp_output.postprocess
    def mlp_output(self, value):
        return rewrap(self, value)


class Moe(Moe, Mlp):
    """Laguna's mixture of experts: a sigmoid router (logits optionally tanh-softcapped) with a selection bias, routed experts and a shared expert, run first.

    ``router_logits`` are the projection's, before the softcap. The mixture
    scales the experts' sum by ``routed_scaling_factor`` before adding the
    shared expert, so ``routed_output`` is that product, bound in this forward
    after the experts have run: the forward is instrumented at build, and
    ``expert_outputs.sum(2) * routed_scaling_factor == routed_output``.
    """

    SCORING = "sigmoid"
    sourced = True

    @TokenEProperty("source.hidden_states_2.output", description="The routed experts' combined output, times routed_scaling_factor")
    def routed_output(self, value) -> Residual:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {LagunaDecoderLayer: Layer, LagunaAttention: Attention, LagunaMLP: Mlp, LagunaSparseMoeBlock: Moe}
