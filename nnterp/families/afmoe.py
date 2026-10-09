"""AFMoE (``AfmoeForCausalLM``, Arcee's Trinity).

Llama's tree with a sandwich block, every sublayer normed before and after::

    h = x + post_attention_layernorm(self_attn(input_layernorm(x)))
    out = h + post_mlp_layernorm(mlp(pre_mlp_layernorm(h)))

What the block adds is each post-norm's output, so ``attention_output`` points at
``post_attention_layernorm`` (which *follows* the attention, as on Gemma-2) and
``mlp_output`` at ``post_mlp_layernorm``; ``pre_mlp_layernorm`` keeps its native
name and ``mlp.input`` is its output. The attention runs the shared eager forward
with q/k norms, the rotary only on its sliding-window (local) blocks, and a
sigmoid gate (``gate_proj``) on the interface's output before ``o_proj``, so
``attention_head_outputs`` are ungated. The first ``num_dense_layers`` blocks have
a dense ``AfmoeMLP``, the rest a mixture (``AfmoeSparseMoeBlock``) returning its
shared expert's output plus the routed experts' sum as one tensor. The shared
expert (``mlp.shared_experts``) is an ``AfmoeMLP`` too, so it is an `Mlp`; its
``mlp_output`` is unavailable, since the block adds the mixture's post-normed sum.
"""

from typing import TYPE_CHECKING

from transformers.models.afmoe.modeling_afmoe import AfmoeAttention, AfmoeDecoderLayer, AfmoeMLP, AfmoeSparseMoeBlock

from ..components import Attention, EProperty, Layer, Mlp, Moe, Residual, RouterLogits, TokenEProperty

if TYPE_CHECKING:
    from nnsight.intervention.envoy import Envoy

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


def _not_a_block_mlp(envoy: "Envoy") -> str | None:
    if envoy.path.rsplit(".", 1)[-1] == "mlp":
        return None
    return "this is the shared expert inside a mixture of experts; what the block adds is the mixture's post-normed output, at layers[i].mlp"


class Layer(Layer):
    """AFMoE's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """AFMoE's attention: the shared eager forward, but what reaches the residual stream is the post-attention norm's output."""

    @EProperty(
        "../post_attention_layernorm.output",
        description="What the attention adds to the residual stream: the post-attention norm's output",
    )
    def attention_output(self, value) -> Residual:
        return value


class Mlp(Mlp):
    """AFMoE's dense MLP or mixture of experts: what reaches the residual stream is the post-MLP norm's output."""

    @EProperty(
        "../post_mlp_layernorm.output",
        description="What the MLP adds to the residual stream: the post-MLP norm's output",
        unavailable=_not_a_block_mlp,
    )
    def mlp_output(self, value) -> Residual:
        return value


class Moe(Moe, Mlp):
    """AFMoE's mixture of experts: a sigmoid router with a selection bias (``expert_bias``, the mixture's), routed experts and a shared expert.

    The router projects with a child ``nn.Linear``, ``gate``: its output is the
    logits. The shared expert runs between the router and the experts.
    """

    SCORING = "sigmoid"

    @TokenEProperty("router.gate.output", description=Moe.router_logits.description)
    def router_logits(self, value) -> RouterLogits:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {AfmoeDecoderLayer: Layer, AfmoeAttention: Attention, AfmoeMLP: Mlp, AfmoeSparseMoeBlock: Moe}
