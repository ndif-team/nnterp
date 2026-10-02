"""GLM-4.5 / 4.6 and GLM-4.5-Air (``Glm4MoeForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, which returns a bare tensor. Attention goes through the shared
eager forward; rotary embeddings cover only ``partial_rotary_factor`` of each head
and ``q_norm``/``k_norm`` run on the heads when ``use_qk_norm`` is set, both before
the interface, so queries and keys are read as the interface sees them. ``head_dim``
is the config's (128 on the real checkpoints, not ``hidden_size // num_heads``).

The first ``first_k_dense_replace`` blocks have a dense MLP, the rest a mixture of
experts. The mixture returns the routed experts' sum plus a shared expert's output
as one tensor, so ``mlp_output`` is everything the block adds after attention; the
shared expert (``mlp.shared_experts``) is a dense MLP of the same class.
"""

from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeAttention, Glm4MoeDecoderLayer, Glm4MoeMLP, Glm4MoeMoE

from ..components import Attention, Layer, Mlp, Moe

MODEL_TYPES = ("glm4_moe",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "gate": "router",
}


class Layer(Layer):
    """GLM-4-MoE's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """GLM-4-MoE's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """GLM-4-MoE's dense MLP or mixture of experts; both return the hidden states (the mixture with its shared expert added), and the residual is added in the block."""


class Moe(Moe, Mlp):
    """GLM-4-MoE's mixture of experts: DeepSeek-V3's sigmoid router, routed experts and a shared expert."""

    SCORING = "sigmoid"


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Glm4MoeDecoderLayer: Layer, Glm4MoeAttention: Attention, Glm4MoeMLP: Mlp, Glm4MoeMoE: Moe}
