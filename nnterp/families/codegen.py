"""CodeGen (``CodeGenForCausalLM``).

``transformer.{wte, drop, h[i].{ln_1, attn, mlp}, ln_f}`` and ``lm_head``. GPT-J's
parallel block: ``ln_1`` feeds both sublayers and ``x + attn + mlp`` is summed in
the block, which returns ``(hidden_states, attn_weights)``. The attention projects
queries, values and keys (in that order) out of one fused ``qkv_proj`` laid out for
four TPU cores, applies rotary embeddings to the first ``rotary_dim`` channels, and
does its own arithmetic in ``_attn``: an fp32 ``q @ k^T``, the mask added, *then*
the division by ``sqrt(head_dim)``, softmax, a cast to the values' dtype,
``attn_dropout``, ``@ v``. The model has no attention implementation but that one,
so ``attn_implementation`` never routes around it and the interior needs no eager
load. ``drop`` has no standard name.
"""

from typing import TYPE_CHECKING

from transformers.models.codegen.modeling_codegen import CodeGenAttention, CodeGenBlock, CodeGenMLP

from ..components import Attention, EProperty, HeadOutputs, Keys, Layer, Mlp, Pattern, Queries, Values, seq_first

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
    """CodeGen's parallel block; returns ``(hidden_states, attn_weights)``, which the base unwraps."""

    returns_tuple = True


class Attention(Attention):
    """CodeGen's attention: the residual is added in the block; the pattern lives in its ``_attn`` method."""

    # The interior lives around the ``_attn`` method call: its arguments are the
    # queries, keys and values heads first, after the rotary embedding (no
    # grouped-query attention, so all three are ``num_heads`` wide), its softmax
    # (an ``nn.Softmax`` built and called in one line, so the call is ``call_0``)
    # takes the masked, scaled fp32 scores, and its first return is the head
    # outputs, heads first. The rotary embedding multiplies by an fp32 buffer, so
    # on a half-precision model the queries (and the keys, without a cache) are
    # served in fp32.

    @EProperty("source.self__attn_0.inputs", select=0, description=Attention.attention_queries.description)
    def attention_queries(self, value) -> Queries:
        return value

    @EProperty("source.self__attn_0.inputs", select=1, description=Attention.attention_keys.description)
    def attention_keys(self, value) -> Keys:
        return value

    @EProperty("source.self__attn_0.inputs", select=2, description=Attention.attention_values.description)
    def attention_values(self, value) -> Values:
        return value

    @EProperty(
        "source.self__attn_0.source.call_0.input",
        description="The attention scores entering the softmax, masked then scaled, in fp32",
    )
    def attention_scores(self, value) -> Pattern:
        return value

    @EProperty("source.self__attn_0.output", select=0, description=Attention.attention_head_outputs.description)
    def attention_head_outputs(self, value) -> HeadOutputs:
        return seq_first(value)

    @attention_head_outputs.postprocess
    def attention_head_outputs(self, value):
        return seq_first(value)

    @EProperty(
        "source.self__attn_0.source.self_attn_dropout_0.output",
        description="The attention pattern the values are mixed with",
    )
    def attention_probabilities(self, value) -> Pattern:
        return value


class Mlp(Mlp):
    """CodeGen's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {CodeGenBlock: Layer, CodeGenAttention: Attention, CodeGenMLP: Mlp}


# -- sizes: what CodeGen's config calls them ----------------------------------------

def intermediate_size(model: "StandardizedTransformer") -> int:
    """The MLP width is ``n_inner``, ``None`` meaning four times the hidden size."""
    return model.config.n_inner or 4 * model.hidden_size
