"""BLOOM (``BloomForCausalLM``).

``transformer.{word_embeddings, word_embeddings_layernorm, h[i].{input_layernorm,
self_attention, post_attention_layernorm, mlp}, ln_f}`` and ``lm_head``. Both
sublayers take the residual as an argument and add it *inside* the module
(``dropout_add``), so the module outputs are residual-stream states; the
contributions are the first argument of each ``dropout_add`` call. The
attention does its own arithmetic whatever ``attn_implementation`` says, so
the pattern is its dropout's output and needs no eager load.
The embedding norm has no standard name.
"""

from typing import TYPE_CHECKING

from transformers.models.bloom.modeling_bloom import BloomAttention, BloomBlock, BloomMLP

from ..components import Attention, EProperty, HeadOutputs, Keys, Layer, Mlp, Pattern, Queries, Residual, Values

if TYPE_CHECKING:
    from ..standardized import StandardizedTransformer

RENAME = {
    "transformer.word_embeddings": "embed_tokens",
    "transformer.h": "layers",
    "transformer.ln_f": "norm",
    "self_attention": "self_attn",
}


class Layer(Layer):
    """BLOOM's block; returns a tuple, which the base unwraps."""

    returns_tuple = True


class Attention(Attention):
    """BLOOM's attention adds the residual inside: the contribution is what enters ``dropout_add``."""

    # ``_reshape`` splits the fused projection into ``(query, key, value)``,
    # heads first (the keys are transposed for the score matmul only later);
    # the scores are the softmax's input after the mask; the head outputs are
    # the ``bmm`` result, ``[batch * heads, seq, head_dim]``.

    @EProperty("source.self__reshape_0.output", select=0, description=Attention.attention_queries.description)
    def attention_queries(self, value) -> Queries:
        return value

    @EProperty("source.self__reshape_0.output", select=1, description=Attention.attention_keys.description)
    def attention_keys(self, value) -> Keys:
        return value

    @EProperty("source.self__reshape_0.output", select=2, description=Attention.attention_values.description)
    def attention_values(self, value) -> Values:
        return value

    @EProperty("source.F_softmax_0.input", description=Attention.attention_scores.description)
    def attention_scores(self, value) -> Pattern:
        return value

    @EProperty("source.torch_bmm_0.output", description=Attention.attention_head_outputs.description)
    def attention_head_outputs(self, value) -> HeadOutputs:
        batch_heads, seq, head_dim = value.shape
        heads = self._module.num_heads
        return value.view(batch_heads // heads, heads, seq, head_dim).transpose(1, 2)

    @attention_head_outputs.postprocess
    def attention_head_outputs(self, value):
        batch, seq, heads, head_dim = value.shape
        return value.transpose(1, 2).reshape(batch * heads, seq, head_dim)

    @EProperty(
        "source.dropout_add_0.input",
        description="What the attention adds to the residual stream: the tensor entering dropout_add",
    )
    def attention_output(self, value) -> Residual:
        return value

    @EProperty(
        "source.self_attention_dropout_0.output",
        description="The attention pattern the values are mixed with",
    )
    def attention_probabilities(self, value) -> Pattern:
        return value


class Mlp(Mlp):
    """BLOOM's MLP adds the residual inside: the contribution is what enters ``dropout_add``."""

    @EProperty(
        "source.dropout_add_0.input",
        description="What the MLP adds to the residual stream: the tensor entering dropout_add",
    )
    def mlp_output(self, value) -> Residual:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {BloomBlock: Layer, BloomAttention: Attention, BloomMLP: Mlp}


# -- sizes: BLOOM's config does not say ---------------------------------------------

def intermediate_size(model: "StandardizedTransformer") -> int:
    """The MLP is four times the hidden size wide; the config has no key for it."""
    return 4 * model.hidden_size
