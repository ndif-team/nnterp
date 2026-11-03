"""GPT-J (``GPTJForCausalLM``).

``transformer.{wte, drop, h[i].{ln_1, attn, mlp}, ln_f}`` and ``lm_head``. A
parallel block: ``ln_1`` feeds both sublayers and ``x + attn + mlp`` is
summed in the block, which returns a tuple. The attention does its own
arithmetic in ``_attn`` rather than the shared eager forward, so the pattern
and the rest of the interior are read around that method call.
"""

import sys
from typing import TYPE_CHECKING

import torch
from transformers.models.gptj.modeling_gptj import GPTJAttention, GPTJBlock, GPTJMLP

from ..components import (
    Attention, EProperty, HeadOutputs, Keys, Layer, Mlp, PARALLEL, Pattern, Queries, Residual, Values, needs_eager, rotary_reason, seq_first,
    unavailable,
)

if TYPE_CHECKING:
    from ..standardized import StandardizedTransformer

RENAME = {
    "transformer.wte": "embed_tokens",
    "transformer.h": "layers",
    "transformer.ln_f": "norm",
    "ln_1": "input_layernorm",
    "attn": "self_attn",
    # The sublayers' interiors: the output projection, the activation and the down projection under their standard names.
    "out_proj": "o_proj",
    "act": "act_fn",
    "fc_out": "down_proj",
}


class Layer(Layer):
    """GPT-J's parallel block; returns ``(hidden_states, present)``, which the base unwraps.

    A parallel block: there is no stream between the sublayers, and the MLP reads the same normed input
    as the attention.
    """

    layer_mid = unavailable(PARALLEL)

    @EProperty("input_layernorm.output", description="The normed stream entering the MLP: the one norm feeds both sublayers")
    def mlp_norm_output(self, value) -> Residual:
        return value


    returns_tuple = True


class Attention(Attention):
    """GPT-J's attention: the residual is added in the block; the pattern lives in its ``_attn`` method."""

    # The rotary runs in the forward itself, tokens before heads: sin and cos
    # gathered from the module's ``embed_positions`` at the position ids, an
    # interleaved rotation (``rotate_every_two``) of each head's leading
    # ``rotary_dim`` channels (the whole head without one). The values before
    # it are the heads-split bindings, served heads first as a view.

    @EProperty("source.query_1.output", description=Attention.attention_queries_prerope.description, unavailable=rotary_reason)
    def attention_queries_prerope(self, value) -> Queries:
        return seq_first(value)

    @attention_queries_prerope.postprocess
    def attention_queries_prerope(self, value):
        return seq_first(value)

    @EProperty("source.key_1.output", description=Attention.attention_keys_prerope.description, unavailable=rotary_reason)
    def attention_keys_prerope(self, value) -> Keys:
        return seq_first(value)

    @attention_keys_prerope.postprocess
    def attention_keys_prerope(self, value):
        return seq_first(value)

    def rotary_embedding(self, x, position_ids):
        """``(sin, cos)`` gathered from the module's sinusoid table at ``position_ids``, in ``x``'s dtype, as the forward does."""
        embed_positions = self._module._get_embed_positions(position_ids)
        repeated = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        sincos = torch.gather(embed_positions, 1, repeated).to(x.dtype)
        return torch.split(sincos, sincos.shape[-1] // 2, dim=-1)

    def apply_rotary(self, x, embedding):
        """The modeling module's ``apply_rotary_pos_emb(tensor, sin, cos)`` on each head's leading ``rotary_dim`` channels, tokens before heads."""
        sin, cos = embedding
        apply = sys.modules[type(self._module).__module__].apply_rotary_pos_emb
        width = self._module.rotary_dim or x.shape[-1]
        tokens = x.reshape(sin.shape[0], -1, *x.shape[-2:]).transpose(1, 2)  # [batch, seq, heads, head_dim]
        rotated = torch.cat((apply(tokens[..., :width], sin, cos), tokens[..., width:]), dim=-1)
        return rotated.transpose(1, 2).reshape(x.shape)

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
