"""Mixtral (``MixtralForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward. The MLP is a sparse mixture of experts
(`Moe`); its router, ``gate``, is aliased ``router``.
"""

from transformers.models.mixtral.modeling_mixtral import MixtralAttention, MixtralDecoderLayer, MixtralSparseMoeBlock

from ..components import Attention, Layer, Moe

MODEL_TYPES = ("mixtral",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "gate": "router",
}


class Layer(Layer):
    """Mixtral's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Mixtral's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Moe):
    """A mixture of experts: the module returns the routed hidden states (a bare tensor on this transformers), so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {MixtralDecoderLayer: Layer, MixtralAttention: Attention, MixtralSparseMoeBlock: Mlp}
