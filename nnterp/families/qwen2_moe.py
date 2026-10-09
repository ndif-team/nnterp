"""Qwen2-MoE (``Qwen2MoeForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward. The MLP is a sparse mixture of experts with a shared expert.

A config with ``mlp_only_layers`` set gives those blocks a dense ``Qwen2MoeMLP`` that no
envoy is keyed to: ``layers[i].mlp`` is a plain Envoy there and ``support()`` reports
"no mlp module on this block". No released checkpoint sets it.
"""

from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeAttention, Qwen2MoeDecoderLayer, Qwen2MoeSparseMoeBlock

from ..components import Attention, Layer, Moe, Residual, TokenEProperty, no_shared_expert

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "gate": "router",
    "shared_expert": "shared_experts",
}


class Layer(Layer):
    """Qwen2-MoE's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Qwen2-MoE's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Moe):
    """A mixture of experts plus a gated shared expert, returned as one bare tensor, so the base `mlp_output` holds.

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
ENVOYS = {Qwen2MoeDecoderLayer: Layer, Qwen2MoeAttention: Attention, Qwen2MoeSparseMoeBlock: Mlp}
