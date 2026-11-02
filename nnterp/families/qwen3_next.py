"""Qwen3-Next (``Qwen3NextForCausalLM``).

Llama's tree with a hybrid block: three blocks in four carry a gated DeltaNet
mixer, ``linear_attn``, the fourth ordinary attention, ``self_attn``
(``config.layer_types``). Each block has one or the other, never both, so on
a linear block every ``self_attn`` value is reported missing and the linear
values live at ``layers[i].linear_attn`` (see `nnterp.LinearAttention`). The MLP is a mixture of experts (a dense MLP class exists for the shared expert).
"""

from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextAttention, Qwen3NextDecoderLayer, Qwen3NextGatedDeltaNet, Qwen3NextMLP, Qwen3NextSparseMoeBlock

from ..components import Attention, Layer, LinearAttention, Mlp, Moe, Residual, TokenEProperty, no_shared_expert

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "gate": "router",
    "shared_expert": "shared_experts",
}


class Layer(Layer):
    """Qwen3-Next's block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Qwen3-Next's softmax attention (one block in four); the shared eager forward, so the base holds."""


class LinearAttention(LinearAttention):
    """Qwen3-Next's gated DeltaNet mixer; transformers' pure-torch chunked rule, so the base holds."""


class Mlp(Mlp):
    """Qwen3-Next's mixture of experts or dense MLP; both return the hidden states, added in the block."""


class Moe(Moe, Mlp):
    """Qwen3-Next's mixture of experts: a softmax router, routed experts and a gated shared expert.

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
ENVOYS = {Qwen3NextDecoderLayer: Layer, Qwen3NextAttention: Attention, Qwen3NextGatedDeltaNet: LinearAttention, Qwen3NextMLP: Mlp, Qwen3NextSparseMoeBlock: Moe}
