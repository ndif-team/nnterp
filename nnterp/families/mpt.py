"""MPT (``MptForCausalLM``).

``transformer.{wte, blocks[i].{norm_1, attn, norm_2, ffn, resid_attn_dropout},
norm_f}`` and ``lm_head``. The block adds the attention's output itself, but
the MLP takes the residual and adds it inside, so its contribution is the
tensor before that add (its dropout's output). The attention does its own
arithmetic whatever ``attn_implementation`` says, so the pattern is its
dropout's output and needs no eager load.
"""

from typing import TYPE_CHECKING

from transformers.models.mpt.modeling_mpt import MptAttention, MptBlock, MptMLP

from ..components import (
    Attention, EProperty, HeadOutputs, Keys, Layer, Mlp, Pattern, Queries, Residual, Values, by_cache, projection_weight, seq_first,
)

if TYPE_CHECKING:
    from ..standardized import StandardizedTransformer

RENAME = {
    "transformer.wte": "embed_tokens",
    "transformer.blocks": "layers",
    "transformer.norm_f": "norm",
    "norm_1": "input_layernorm",
    "norm_2": "post_attention_layernorm",
    "attn": "self_attn",
    "ffn": "mlp",
    # The sublayers' interiors: the output projection, the activation and the down projection under their standard names.
    "out_proj": "o_proj",
    "act": "act_fn",
}


class Layer(Layer):
    """MPT's block; returns a tuple, which the base unwraps."""

    returns_tuple = True


#: Where the queries, keys and values are, by whether the call has a cache: the keys and values are
#: ``past_key_values.update``'s return where it has one, else the reshaped bindings; the queries are their binding.
QUERIES = by_cache(("source.query_states_0.output", None), ("source.query_states_0.output", None))
KEYS = by_cache(("source.past_key_values_update_0.output", 0), ("source.key_states_0.output", None))
VALUES = by_cache(("source.past_key_values_update_0.output", 1), ("source.value_states_0.output", None))


class Attention(Attention):
    """MPT's attention: the residual is added in the block; the pattern is its own dropout."""

    def no_rotary(self) -> str | None:
        return "no rotary embedding: MPT biases its scores with ALiBi"

    # The three ``*_states`` bindings are the projections reshaped heads first
    # (no grouped-query attention). The forward then updates the cache itself,
    # ``past_key_values.update(key_states, value_states)``, and computes with
    # what that returns, every position so far, so the keys and values are the
    # update's return where the call has a cache and the bindings where it has
    # none (`by_cache`; the queries ask too, so reading them first leaves the
    # keys able to). The scores are the softmax's input after the mask,
    # upcast to float32; the head outputs are the second matmul, heads first.

    @EProperty(QUERIES[0], select=QUERIES[1], description=Attention.attention_queries.description)
    def attention_queries(self, value) -> Queries:
        return value

    @EProperty(KEYS[0], select=KEYS[1], description=Attention.attention_keys.description)
    def attention_keys(self, value) -> Keys:
        """The keys the attention computes with: the cache's, every position so far, where the call has a cache."""
        return value

    @EProperty(VALUES[0], select=VALUES[1], description=Attention.attention_values.description)
    def attention_values(self, value) -> Values:
        """The values the attention computes with: the cache's, every position so far, where the call has a cache."""
        return value

    @EProperty("source.nn_functional_softmax_0.input", description=Attention.attention_scores.description)
    def attention_scores(self, value) -> Pattern:
        return value

    @EProperty("source.torch_matmul_1.output", description=Attention.attention_head_outputs.description)
    def attention_head_outputs(self, value) -> HeadOutputs:
        return seq_first(value)

    @attention_head_outputs.postprocess
    def attention_head_outputs(self, value):
        return seq_first(value)

    @EProperty(
        "source.nn_functional_dropout_0.output",
        description="The attention pattern the values are mixed with",
    )
    def attention_probabilities(self, value) -> Pattern:
        return value

    def value_weight(self, kv_head):
        """``Wqkv``'s rows are every query, key, then value; a ``clip_qkv`` checkpoint clamps the projection's output after it."""
        return projection_weight(self, "Wqkv").unflatten(0, (3, self.num_heads, self.head_dim))[2, kv_head]


class Mlp(Mlp):
    """MPT's MLP adds the residual inside: the contribution is its dropout's output, before the add."""

    @EProperty(
        "source.F_dropout_0.output",
        description="What the MLP adds to the residual stream: the tensor before the residual is added inside",
    )
    def mlp_output(self, value) -> Residual:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {MptBlock: Layer, MptAttention: Attention, MptMLP: Mlp}


# -- sizes: what MPT's config calls them --------------------------------------------

def intermediate_size(model: "StandardizedTransformer") -> int:
    """The MLP width is ``expansion_ratio`` times the hidden size (transformers builds ``4 * hidden_size``; equal on the released checkpoints)."""
    return int(model.config.expansion_ratio * model.hidden_size)
