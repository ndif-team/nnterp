"""dots.llm1 (``Dots1ForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward.
The attention norms its queries and keys per head (``q_norm``, ``k_norm``)
before the rotary embedding. The first ``first_k_dense_replace`` blocks have a
dense MLP, the rest a mixture of experts that adds a shared expert
(``mlp.shared_experts``, itself a ``Dots1MLP``) before it returns; the experts
are ``moe_intermediate_size`` wide.
"""

from transformers.models.dots1.modeling_dots1 import Dots1Attention, Dots1DecoderLayer, Dots1MLP, Dots1MoE

from ..components import Attention, Layer, Mlp, Moe

MODEL_TYPES = ("dots1",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "gate": "router",
}


class Layer(Layer):
    """dots.llm1's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """dots.llm1's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """dots.llm1's dense MLP or mixture of experts; both return the hidden states (the mixture with its shared expert added), and the residual is added in the block."""


class Moe(Moe, Mlp):
    """dots.llm1's mixture of experts: DeepSeek-V3's sigmoid router, routed experts and a shared expert."""

    SCORING = "sigmoid"


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Dots1DecoderLayer: Layer, Dots1Attention: Attention, Dots1MLP: Mlp, Dots1MoE: Moe}
