"""GPT-NeoX-Japanese (``GPTNeoXJapaneseForCausalLM``).

``gpt_neox_japanese.{embed_in, layers[i].{input_layernorm, post_attention_layernorm,
attention, mlp}, final_layer_norm, rotary_emb}`` and ``embed_out``, the head. GPT-NeoX's
names, but a sequential block, not a parallel one: ``post_attention_layernorm``
normalizes the stream after the attention has been added. The block adds each
sublayer through ``bias_dropout_add(x, bias, residual)``, ``residual + dropout(x +
bias)``, and returns ``(hidden_states, attn_weights)``. The attention returns
``(attn_output, attn_weights, dense_bias)``: its output projection has no bias, and
on the last block only a separate ``dense_bias`` is handed to the block to add, so
what the attention adds is its output plus that bias: a sum served as
``attention_output``, with an edit carried back to the module's output minus the
bias. The attention does its own arithmetic in ``_attn`` whatever
``attn_implementation`` says (the model has no other): a ``baddbmm`` scaled by
``1/sqrt(head_dim)``, the mask added, softmax, ``attention_dropout``, a cast to the
values' dtype, ``@ v``; the interior needs no eager load.
"""

from typing import TYPE_CHECKING

import torch
from transformers.models.gpt_neox_japanese.modeling_gpt_neox_japanese import (
    GPTNeoXJapaneseAttention, GPTNeoXJapaneseLayer, GPTNeoXJapaneseMLP,
)

from ..components import (
    Attention, EProperty, HeadOutputs, Keys, Layer, Mlp, Pattern, Queries, Residual, Values, first_tensor, rewrap,
    seq_first,
)

if TYPE_CHECKING:
    from ..standardized import StandardizedTransformer

MODEL_TYPES = ("gpt_neox_japanese",)

RENAME = {
    "gpt_neox_japanese.embed_in": "embed_tokens",
    "gpt_neox_japanese.layers": "layers",
    "gpt_neox_japanese.final_layer_norm": "norm",
    "embed_out": "lm_head",
    "attention": "self_attn",
}


class Layer(Layer):
    """GPT-NeoX-Japanese's sequential block; returns ``(hidden_states, attn_weights)``, which the base unwraps."""

    returns_tuple = True


class Attention(Attention):
    """GPT-NeoX-Japanese's attention: the bias and the residual are added in the block; the pattern lives in its ``_attn`` method."""

    # The interior lives around the ``_attn`` method call: its arguments are the
    # queries, keys and values heads first, after the rotary embedding and the
    # cache update (no grouped-query attention, so all three are ``num_heads``
    # wide), its softmax takes the scaled, masked scores, and its first return
    # is the head outputs, heads first.

    @EProperty("source.self__attn_0.inputs", select=0, description=Attention.attention_queries.description)
    def attention_queries(self, value) -> Queries:
        return value

    @EProperty("source.self__attn_0.inputs", select=1, description=Attention.attention_keys.description)
    def attention_keys(self, value) -> Keys:
        return value

    @EProperty("source.self__attn_0.inputs", select=2, description=Attention.attention_values.description)
    def attention_values(self, value) -> Values:
        return value

    @EProperty("source.self__attn_0.source.nn_functional_softmax_0.input", description=Attention.attention_scores.description)
    def attention_scores(self, value) -> Pattern:
        return value

    @EProperty("source.self__attn_0.output", select=0, description=Attention.attention_head_outputs.description)
    def attention_head_outputs(self, value) -> HeadOutputs:
        return seq_first(value)

    @attention_head_outputs.postprocess
    def attention_head_outputs(self, value):
        return seq_first(value)

    @EProperty(
        "source.self__attn_0.source.self_attention_dropout_0.output",
        description="The attention pattern the values are mixed with",
    )
    def attention_probabilities(self, value) -> Pattern:
        return value

    @EProperty(
        key="output",
        description="What the attention adds to the residual stream: its output plus, on the last block, the dense_bias the block adds",
    )
    def attention_output(self, value) -> Residual:
        bias = self._module.dense_bias
        return first_tensor(value) if bias is None else first_tensor(value) + bias

    @attention_output.postprocess
    def attention_output(self, value):
        bias = self._module.dense_bias
        return rewrap(self, value if bias is None else value - bias)

    @attention_output.transform
    def attention_output(self, value, raw):
        # Fires on the model side, after the read. Without a bias the read is
        # the module's own tensor and in-place edits are already in ``raw``.
        # With one it is a sum; the module's output changes only when the sum
        # was edited, so a plain read leaves the model bit-identical.
        bias = self._module.dense_bias
        if bias is None or torch.equal(value, raw[0] + bias):
            return raw
        return (value - bias, *raw[1:])


class Mlp(Mlp):
    """GPT-NeoX-Japanese's MLP; the residual is added in the block with no bias, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {GPTNeoXJapaneseLayer: Layer, GPTNeoXJapaneseAttention: Attention, GPTNeoXJapaneseMLP: Mlp}


# -- sizes: what GPT-NeoX-Japanese's config calls them ------------------------------

def intermediate_size(model: "StandardizedTransformer") -> int:
    """The MLP width is ``hidden_size * intermediate_multiple_size``."""
    return int(model.hidden_size * model.config.intermediate_multiple_size)
