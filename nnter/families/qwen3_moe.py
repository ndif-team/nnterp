"""Qwen3-MoE (``Qwen3MoeForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward. The MLP is a sparse mixture of experts; ``head_dim`` is the config's.
"""

from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention, Qwen3MoeDecoderLayer, Qwen3MoeSparseMoeBlock

from ..components import Attention, Layer, Moe

MODEL_TYPES = ("qwen3_moe",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "gate": "router",
}


class Layer(Layer):
    """Qwen3-MoE's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Qwen3-MoE's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Moe):
    """A mixture of experts: the module returns the routed hidden states (a bare tensor on this transformers), so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Qwen3MoeDecoderLayer: Layer, Qwen3MoeAttention: Attention, Qwen3MoeSparseMoeBlock: Mlp}
