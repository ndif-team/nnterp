"""OLMoE (``OlmoeForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward. The attention
norms its queries and keys (``q_norm``, ``k_norm``) across all heads before the
rotary embedding, and optionally clamps them (``clip_qkv``); both happen before
the interface call, so the queries and keys are read after them. Every MLP is a
sparse mixture of experts that returns the routed hidden states as a bare tensor;
its experts are ``intermediate_size`` wide.
"""

from transformers.models.olmoe.modeling_olmoe import OlmoeAttention, OlmoeDecoderLayer, OlmoeSparseMoeBlock

from ..components import Attention, Layer, Moe

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "gate": "router",
}


class Layer(Layer):
    """OLMoE's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """OLMoE's attention; the query and key norms run before the shared eager forward and the residual is added in the block, so the base holds."""


class Mlp(Moe):
    """A mixture of experts: the module returns the routed hidden states as a bare tensor, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {OlmoeDecoderLayer: Layer, OlmoeAttention: Attention, OlmoeSparseMoeBlock: Mlp}
