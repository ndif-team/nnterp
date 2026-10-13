"""Phi-3.5-MoE (``PhimoeForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward.
The norms are ``nn.LayerNorm``. Every block's MLP is a sparse mixture of experts
that returns the routed hidden states as a bare tensor; the experts are
``intermediate_size`` wide.
"""

from transformers.models.phimoe.modeling_phimoe import PhimoeAttention, PhimoeDecoderLayer, PhimoeSparseMoeBlock

from ..components import Attention, Layer, Moe, RouterLogits, TokenEProperty

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """Phi-3.5-MoE's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Phi-3.5-MoE's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Moe):
    """A mixture of experts: the module returns the routed hidden states as a bare tensor, so the base holds.

    The router is an ``nn.Linear`` subclass whose forward calls its parent's
    (``super().forward``), so the logits are that call's output; sparsemixer
    turns them into the weights.
    """

    SCORING = "sparsemixer"

    @TokenEProperty("router.source.forward_0.output", description=Moe.router_logits.description)
    def router_logits(self, value) -> RouterLogits:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {PhimoeDecoderLayer: Layer, PhimoeAttention: Attention, PhimoeSparseMoeBlock: Mlp}
