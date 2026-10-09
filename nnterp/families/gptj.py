"""GPT-J (``GPTJForCausalLM``).

``transformer.{wte, drop, h[i].{ln_1, attn, mlp}, ln_f}`` and ``lm_head``. A
parallel block: ``ln_1`` feeds both sublayers and ``x + attn + mlp`` is
summed in the block, which returns a tuple. The attention does its own
arithmetic in ``_attn`` rather than the shared eager forward, so the pattern
and the rest of the interior are read around that method call.
"""

from typing import TYPE_CHECKING

from transformers.models.gptj.modeling_gptj import GPTJAttention, GPTJBlock, GPTJMLP

from ..components import (
    Attention, EProperty, HeadOutputs, Keys, Layer, Mlp, Pattern, Queries, Values, needs_eager, seq_first,
)

if TYPE_CHECKING:
    from ..standardized import StandardizedTransformer

RENAME = {
    "transformer.wte": "embed_tokens",
    "transformer.h": "layers",
    "transformer.ln_f": "norm",
    "ln_1": "input_layernorm",
    "attn": "self_attn",
}


class Layer(Layer):
    """GPT-J's parallel block; returns ``(hidden_states, present)``, which the base unwraps."""

    returns_tuple = True


class Attention(Attention):
    """GPT-J's attention: the residual is added in the block; the pattern lives in its ``_attn`` method."""

    # The interior lives around the ``_attn`` method call: its arguments are the
    # queries, keys and values after rotary embeddings (no grouped-query
    # attention here, so keys and values are ``num_heads`` wide), its softmax
    # takes the masked scores, and its first return is the head outputs, heads
    # first.

    @EProperty("source.self__attn_0.inputs", select=0, description=Attention.attention_queries.description, unavailable=needs_eager)
    def attention_queries(self, value) -> Queries:
        return value

    @EProperty("source.self__attn_0.inputs", select=1, description=Attention.attention_keys.description, unavailable=needs_eager)
    def attention_keys(self, value) -> Keys:
        return value

    @EProperty("source.self__attn_0.inputs", select=2, description=Attention.attention_values.description, unavailable=needs_eager)
    def attention_values(self, value) -> Values:
        return value

    @EProperty("source.self__attn_0.source.nn_functional_softmax_0.input", description=Attention.attention_scores.description, unavailable=needs_eager)
    def attention_scores(self, value) -> Pattern:
        return value

    @EProperty("source.self__attn_0.output", select=0, description=Attention.attention_head_outputs.description, unavailable=needs_eager)
    def attention_head_outputs(self, value) -> HeadOutputs:
        return seq_first(value)

    @attention_head_outputs.postprocess
    def attention_head_outputs(self, value):
        return seq_first(value)

    @EProperty(
        "source.self__attn_0.source.self_attn_dropout_0.output",
        description="The attention pattern the values are mixed with",
        unavailable=needs_eager,
    )
    def attention_probabilities(self, value) -> Pattern:
        return value


class Mlp(Mlp):
    """GPT-J's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {GPTJBlock: Layer, GPTJAttention: Attention, GPTJMLP: Mlp}


# -- sizes: what GPT-J's config calls them ------------------------------------------

def intermediate_size(model: "StandardizedTransformer") -> int:
    """The MLP width is ``n_inner``, ``None`` meaning four times the hidden size."""
    return model.config.n_inner or 4 * model.hidden_size
