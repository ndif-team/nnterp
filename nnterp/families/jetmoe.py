"""JetMoE (``JetMoeForCausalLM``).

Llama's tree with the attention under its own name: ``model.{embed_tokens,
layers[i].{input_layernorm, self_attention, post_attention_layernorm, mlp}, norm}``
and ``lm_head``, the residual added in the block, which returns a bare tensor.
``self_attention`` is aliased ``self_attn``.

The attention is a mixture of attention heads: a router picks
``num_experts_per_tok`` experts per token, each expert projects that token's
queries for ``num_key_value_heads`` heads (``experts.map``) and later projects their
outputs back (``experts.reduce``, in place of an ``o_proj``). Keys and values come
from one shared ``kv_proj`` and are tiled (``repeat``, not interleaved) to every
query head before the shared eager forward, so the interface sees ``num_heads``
key/value heads and interface head ``h`` is routing slot ``h // num_kv_heads`` over
kv head ``h % num_kv_heads``: which expert's parameters a head's query came from
varies by token. The attention returns ``(attn_output, attn_weights,
router_logits)``. The MLP is a mixture of experts (`Moe`) that returns the routed hidden
states plus a bias as one tensor.

The MLP's mixture has no experts module: its router (``JetMoeTopKGating``) takes a
top-k of the logits of a child ``nn.Linear`` (``router.layer``), softmaxes the top-k,
and sorts the slots by expert; the mixture then runs ``input_linear`` and
``output_linear`` over the sorted slots. So ``expert_indices`` and ``expert_weights``
are the router's top-k indices and gates, ``[tokens, top_k]`` in token order, before
the sort; ``routed_output`` is the mixture's routed sum before ``+ bias``; and the
per-slot outputs, sorted by expert, are unavailable.
"""

from transformers.models.jetmoe.modeling_jetmoe import JetMoeAttention, JetMoeDecoderLayer, JetMoeMoE

from ..components import (
    Attention, ExpertIndices, ExpertWeights, Layer, Moe, Residual, RouterLogits, TokenEProperty, unavailable,
)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "self_attention": "self_attn",
}


class Layer(Layer):
    """JetMoE's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """JetMoE's mixture of attention heads; the shared eager forward, a 3-tuple whose first element is the contribution, and the residual added in the block, so the base holds."""


class Mlp(Moe):
    """A mixture of experts: the module returns the routed hidden states (plus its bias) as a bare tensor, so the base `mlp_output` holds; the width is spelled its own way.

    The routing values are the router's own operations and ``routed_output`` an
    operation of this forward read after the router has run, so the forward is
    instrumented at build. Read order: ``router_logits``, ``expert_indices``,
    ``expert_weights``, ``routed_output``.
    """

    SCORING = "topk_softmax"
    sourced = True

    @TokenEProperty("router.layer.output", description=Moe.router_logits.description)
    def router_logits(self, value) -> RouterLogits:
        return value

    @TokenEProperty("router.source.top_k_gates_0.output", description=Moe.expert_weights.description)
    def expert_weights(self, value) -> ExpertWeights:
        """The router's softmax over the top-k logits, ``[batch, seq, top_k]``, in token order before the router sorts the slots by expert."""
        return value

    @TokenEProperty("router.source.logits_topk_0.output", select=1, description=Moe.expert_indices.description)
    def expert_indices(self, value) -> ExpertIndices:
        """The router's top-k indices, ``[batch, seq, top_k]``, in token order before the router sorts the slots by expert."""
        return value

    expert_outputs = unavailable("JetMoE's mixture runs its experts over the slots sorted by expert; no tensor holds them in token order")

    @TokenEProperty("source.layer_output_1.output", description="The routed experts' combined output, before the mixture's bias")
    def routed_output(self, value) -> Residual:
        """The routed sum in the residual's shape, before ``+ bias``: ``mlp.output - routed_output`` is the bias."""
        return value

    @property
    def intermediate_size(self) -> int:
        """One expert's width: the module calls it ``hidden_size`` (its ``input_size`` is the model's)."""
        return self._module.hidden_size


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {JetMoeDecoderLayer: Layer, JetMoeAttention: Attention, JetMoeMoE: Mlp}
