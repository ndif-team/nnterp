"""EXAONE 4.0 (``Exaone4ForCausalLM``).

Llama's names plus a post-norm block, OLMo-2's layout: every sublayer is normed
after, ``x + post_attention_layernorm(attn(x))`` and then
``+ post_feedforward_layernorm(mlp(x))``. What the block adds is the post-norm's
output, not the module's, so the contributions point at the sibling norms.
EXAONE 4.0 has only the post-norms: no ``input_layernorm`` exists to alias.
The attention runs the shared eager forward with per-head q/k norms. A checkpoint
with a ``sliding_window`` mixes sliding-window layers, which apply the rotary, and
full-attention layers, which apply none (NoPE); both reach the interface the same
way, so the values are the same on every block.
"""

from transformers.models.exaone4.modeling_exaone4 import Exaone4Attention, Exaone4DecoderLayer, Exaone4MLP

from ..components import Attention, EProperty, Layer, Mlp, Residual

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """EXAONE-4's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """EXAONE-4's attention: the shared eager forward, but what reaches the residual stream is the post-attention norm's output."""

    @EProperty(
        "../post_attention_layernorm.output",
        description="What the attention adds to the residual stream: the post-attention norm's output",
    )
    def attention_output(self, value) -> Residual:
        return value


class Mlp(Mlp):
    """EXAONE-4's MLP: what reaches the residual stream is the post-feedforward norm's output."""

    @EProperty(
        "../post_feedforward_layernorm.output",
        description="What the MLP adds to the residual stream: the post-feedforward norm's output",
    )
    def mlp_output(self, value) -> Residual:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Exaone4DecoderLayer: Layer, Exaone4Attention: Attention, Exaone4MLP: Mlp}
