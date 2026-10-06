"""MiniMax-M2 (``MiniMaxM2ForCausalLM``): MiniMax-M2, M2.5 and M2.7.

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block with no scaling, attention through the shared eager forward.
The attention norms its queries and keys across the whole projection
(``q_norm``/``k_norm`` over ``heads * head_dim``) before the head split; the
queries and keys are read where they enter the interface, after the norms and
the rotary, so nothing changes. Every block's MLP is a sparse mixture of experts
with a sigmoid router and a correction bias; it returns the routed hidden states
as a bare tensor. ``intermediate_size`` is the experts' width and ``head_dim`` is
the config's.
"""

from transformers.models.minimax_m2.modeling_minimax_m2 import (
    MiniMaxM2Attention,
    MiniMaxM2DecoderLayer,
    MiniMaxM2SparseMoeBlock,
)

from ..components import Attention, Layer, Moe

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "gate": "router",
}


class Layer(Layer):
    """MiniMax-M2's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """MiniMax-M2's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Moe):
    """A mixture of experts: the module returns the routed hidden states as a bare tensor, so the base holds."""

    SCORING = "sigmoid"


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {MiniMaxM2DecoderLayer: Layer, MiniMaxM2Attention: Attention, MiniMaxM2SparseMoeBlock: Mlp}
