"""ERNIE 4.5 MoE (``Ernie4_5_MoeForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward.
Blocks from ``moe_layer_start_index`` to ``moe_layer_end_index``, every
``moe_layer_interval``-th, have a mixture of experts (with optional shared experts,
added before it returns); the others a dense MLP. The experts are
``moe_intermediate_size`` wide.
"""

from transformers.models.ernie4_5_moe.modeling_ernie4_5_moe import (
    Ernie4_5_MoeAttention,
    Ernie4_5_MoeDecoderLayer,
    Ernie4_5_MoeMLP,
    Ernie4_5_MoeSparseMoeBlock,
)

from ..components import Attention, Layer, Mlp, Moe

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "gate": "router",
}


class Layer(Layer):
    """ERNIE 4.5 MoE's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """ERNIE 4.5 MoE's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """ERNIE 4.5's dense MLP or mixture of experts; both return the hidden states (the mixture with its shared experts added), and the residual is added in the block."""


class Moe(Moe, Mlp):
    """ERNIE 4.5's mixture of experts: a softmax router (a selection bias, ``moe_statics``), routed experts and optional shared experts, run first."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Ernie4_5_MoeDecoderLayer: Layer, Ernie4_5_MoeAttention: Attention, Ernie4_5_MoeMLP: Mlp, Ernie4_5_MoeSparseMoeBlock: Moe}
