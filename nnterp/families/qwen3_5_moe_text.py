"""Qwen3.5-MoE / 3.6-MoE text (``Qwen3_5MoeForCausalLM``, model_type ``qwen3_5_moe_text``).

Llama's tree with a hybrid block: three blocks in four carry a gated DeltaNet
mixer, ``linear_attn``, the fourth ordinary attention, ``self_attn``
(``config.layer_types``). Each block has one or the other, never both, so on
a linear block every ``self_attn`` value is reported missing and the linear
values live at ``layers[i].linear_attn`` (see `nnterp.LinearAttention`). The MLP is a mixture of experts.
"""

from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeAttention, Qwen3_5MoeDecoderLayer, Qwen3_5MoeGatedDeltaNet, Qwen3_5MoeMLP, Qwen3_5MoeSparseMoeBlock

from ..components import Attention, Layer, LinearAttention, Mlp, Moe, Residual, TokenEProperty, no_shared_expert

MODEL_TYPES = ("qwen3_5_moe_text",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "gate": "router",
    "shared_expert": "shared_experts",
}


class Layer(Layer):
    """Qwen3.5-MoE's block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Qwen3.5-MoE's softmax attention (one block in four); the shared eager forward, so the base holds."""


class LinearAttention(LinearAttention):
    """Qwen3.5-MoE's gated DeltaNet mixer; transformers' pure-torch chunked rule, so the base holds."""


class Mlp(Mlp):
    """Qwen3.5-MoE's mixture of experts or dense MLP; both return the hidden states, added in the block."""


class Moe(Moe, Mlp):
    """Qwen3.5-MoE's mixture of experts: a softmax router, routed experts and a gated shared expert.

    The shared expert's contribution is its output times a sigmoid gate,
    ``F.sigmoid(shared_expert_gate(x)) * shared_expert(x)``, bound in this
    forward after the experts have run, so the forward is instrumented at build.
    """

    sourced = True

    @TokenEProperty("source.shared_expert_output_1.output", description=Moe.shared_expert_output.description, unavailable=no_shared_expert)
    def shared_expert_output(self, value) -> Residual:
        """The shared expert's output times its sigmoid gate (``shared_expert_gate``), the product the mixture adds."""
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Qwen3_5MoeDecoderLayer: Layer, Qwen3_5MoeAttention: Attention, Qwen3_5MoeGatedDeltaNet: LinearAttention, Qwen3_5MoeMLP: Mlp, Qwen3_5MoeSparseMoeBlock: Moe}
