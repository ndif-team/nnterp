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

CodeGen2 checkpoints (``Salesforce/codegen2-*``) declare ``model_type`` ``codegen``.
The 1B and 3.7B checkpoints' own code lays ``qkv_proj`` out in eight groups
(``mp_num = 8``); loaded through transformers' class, which splits it in four, their
queries, keys and values are cut wrong and nothing reports it. The 7B and 16B
checkpoints' code uses four groups, as transformers' class does.
"""

from typing import TYPE_CHECKING

import torch
from transformers.models.codegen.modeling_codegen import CodeGenAttention, CodeGenBlock, CodeGenMLP

from ..components import (
    Attention, EProperty, HeadOutputs, Keys, Layer, Mlp, PARALLEL, Pattern, Queries, Residual, Values, projection_weight, rotary_reason, seq_first,
    unavailable,
)
from .gptj import Attention as GPTJAttention

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
    """CodeGen's parallel block; returns ``(hidden_states, attn_weights)``, which the base unwraps.

    A parallel block: there is no stream between the sublayers, and the MLP reads the same normed input
    as the attention.
    """

    layer_mid = unavailable(PARALLEL)

    @EProperty("input_layernorm.output", description="The normed stream entering the MLP: the one norm feeds both sublayers")
    def mlp_norm_output(self, value) -> Residual:
        return value


    returns_tuple = True


class Attention(Attention):
    """CodeGen's attention: the residual is added in the block; the pattern lives in its ``_attn`` method."""

    # GPT-J's rotary: tokens before heads, an interleaved rotation of each
    # head's leading ``rotary_dim`` channels, so the values before it and the
    # arithmetic are GPT-J's; sin and cos are indexed from the fp32 table and
    # not cast, which is why the rotated queries come out fp32.

    @EProperty("source.query_0.output", description=Attention.attention_queries_prerope.description, unavailable=rotary_reason)
    def attention_queries_prerope(self, value) -> Queries:
        return seq_first(value)

    @attention_queries_prerope.postprocess
    def attention_queries_prerope(self, value):
        return seq_first(value)

    @EProperty("source.key_0.output", description=Attention.attention_keys_prerope.description, unavailable=rotary_reason)
    def attention_keys_prerope(self, value) -> Keys:
        return seq_first(value)

    @attention_keys_prerope.postprocess
    def attention_keys_prerope(self, value):
        return seq_first(value)

    def rotary_embedding(self, x, position_ids):
        """``(sin, cos)`` indexed from the module's fp32 sinusoid table at ``position_ids``, as the forward does."""
        sincos = self._module.embed_positions.to(position_ids.device)[position_ids]
        return torch.split(sincos, sincos.shape[-1] // 2, dim=-1)

    apply_rotary = GPTJAttention.apply_rotary

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

    def value_weight(self, kv_head):
        """``qkv_proj``'s rows are four parts (the forward's ``mp_num``), each a quarter of the heads' queries, values, then keys."""
        per_part = self.num_heads // 4
        weight = projection_weight(self, "qkv_proj").unflatten(0, (4, 3, per_part, self.head_dim))
        return weight[kv_head // per_part, 1, kv_head % per_part]


class Mlp(Mlp):
    """CodeGen's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {CodeGenBlock: Layer, CodeGenAttention: Attention, CodeGenMLP: Mlp}


# -- sizes: what CodeGen's config calls them ----------------------------------------

def intermediate_size(model: "StandardizedTransformer") -> int:
    """The MLP width is ``n_inner``, ``None`` meaning four times the hidden size."""
    return model.config.n_inner or 4 * model.hidden_size
