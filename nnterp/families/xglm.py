"""XGLM (``XGLMForCausalLM``).

``model.{embed_tokens, embed_positions, layers[i].{self_attn, self_attn_layer_norm,
fc1, fc2, final_layer_norm}, layer_norm}`` and ``lm_head``. ``embed_tokens`` is a
scaled embedding (times ``sqrt(d_model)`` under ``scale_embedding``), so
``token_embeddings`` is the scaled lookup; the sinusoidal positions are added
after it, outside any module with a standard name. As on OPT there is no MLP
module: ``fc1`` and ``fc2`` sit on the block, so ``mlp`` and ``mlp_output`` do not
exist here and `StandardizedTransformer.support` says so. The block's own
``final_layer_norm`` is the pre-MLP norm and keeps its native name. The block
returns a bare tensor. The attention does its own arithmetic whatever
``attn_implementation`` says (the model has no other): queries scaled by
``1/sqrt(head_dim)`` as they are projected, queries, keys and values flattened to
``[batch * num_heads, seq, head_dim]`` for ``torch.bmm``, the mask added and clamped,
softmax, dropout, ``bmm`` with the values; the interior needs no eager load. nnsight
narrows a tensor to an invoke's rows only when its leading axis is the batch, so
the queries, scores, pattern and head outputs are read where the forward still
holds them batch first, and each invoke of a trace reads and edits its own rows of
them. The keys and values are read at the ``reshape`` that flattens them, after the
cache update, the one operation that holds them with or without a cache: each
invoke reads, and an edit reaches, the whole batch's.
"""

from typing import TYPE_CHECKING

import torch
from transformers.models.xglm.modeling_xglm import XGLMAttention, XGLMDecoderLayer

from ..components import Attention, EProperty, HeadOutputs, Keys, Layer, Mlp, Pattern, Queries, Residual, Values

if TYPE_CHECKING:
    from nnsight import Envoy

    from ..standardized import StandardizedTransformer

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.layer_norm": "norm",
    "self_attn_layer_norm": "input_layernorm",
    # The sublayers' interiors: the output projection, the activation and the down projection under their standard names.
    "out_proj": "o_proj",
}


def _dropping(envoy: "Envoy") -> bool:
    """Whether the attention dropout acts on this call: in training, with a dropout probability."""
    return envoy._module.training and envoy._module.dropout > 0


def _pattern(envoy: "Envoy") -> str:
    """The softmax's output viewed heads first, which the dropout leaves as it is outside training; in training the
    dropout's output, which is only flat."""
    return f"source.{'nn_functional_dropout_0' if _dropping(envoy) else 'attn_weights_reshaped_0'}.output"


_pattern.__name__ = "attn_weights_reshaped_0|nn_functional_dropout_0"


def _heads_first(envoy: "Envoy", value: torch.Tensor) -> torch.Tensor:
    """A ``[batch * num_heads, ...]`` tensor viewed as ``[batch, num_heads, ...]``."""
    heads = envoy._module.num_heads
    return value.view(value.shape[0] // heads, heads, *value.shape[1:])


def _flat(value: torch.Tensor) -> torch.Tensor:
    """A ``[batch, num_heads, ...]`` tensor back to the ``[batch * num_heads, ...]`` the forward holds."""
    return value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])


class Layer(Layer):
    """XGLM's decoder block; returns a bare tensor, so the base holds.

    The pre-MLP norm is the block's own ``final_layer_norm``, so ``layer_mid`` and ``mlp_norm_output``
    read there; there is no ``mlp`` module, so ``mlp_input`` does not exist.
    """

    @EProperty("final_layer_norm.input", description="The residual stream after the attention add")
    def layer_mid(self, value) -> Residual:
        return value

    @EProperty("final_layer_norm.output", description="The normed stream entering the feed-forward path")
    def mlp_norm_output(self, value) -> Residual:
        return value



class Attention(Attention):
    """XGLM's attention: the residual is added in the block; the interior on its own ``bmm`` arithmetic."""

    def no_rotary(self) -> str | None:
        return "no rotary embedding: XGLM adds sinusoidal absolute position embeddings to the stream"

    # The forward flattens queries, keys and values to ``[batch * num_heads,
    # ...]`` for ``bmm``, a layout nnsight cannot narrow to an invoke's rows, so
    # the queries, scores, pattern and head outputs are read where the forward
    # still holds them batch first: the queries at the scaled projection (viewed
    # heads first), the scores at the clamp of the masked scores
    # (``attn_weights_2``), the pattern at the softmax's output viewed heads
    # first (the dropout's output, flat, in training), the head outputs at the
    # transpose before the merge. The keys and values are the ``reshape`` calls
    # that flatten them after the cache update, viewed back to heads first and
    # reshaped back on a write: the update runs only where the call has a cache,
    # and the ``reshape`` is the one operation after it that holds them either
    # way, so each invoke reads the whole batch's. Every read is the tensor the
    # forward holds, or a view of it, so in-place edits reach the model.

    @EProperty("source.query_states_0.output", description="The queries entering attention, already scaled by 1/sqrt(head_dim)")
    def attention_queries(self, value) -> Queries:
        batch, seq, _ = value.shape
        return value.view(batch, seq, self._module.num_heads, -1).transpose(1, 2)

    @attention_queries.postprocess
    def attention_queries(self, value):
        return value.transpose(1, 2).flatten(2)

    @EProperty("source.key_states_reshape_0.output", description=Attention.attention_keys.description)
    def attention_keys(self, value) -> Keys:
        return _heads_first(self, value)

    @attention_keys.postprocess
    def attention_keys(self, value):
        return _flat(value)

    @EProperty("source.value_states_reshape_0.output", description=Attention.attention_values.description)
    def attention_values(self, value) -> Values:
        return _heads_first(self, value)

    @attention_values.postprocess
    def attention_values(self, value):
        return _flat(value)

    @EProperty("source.attn_weights_2.output", description=Attention.attention_scores.description)
    def attention_scores(self, value) -> Pattern:
        """The masked scores clamped at the dtype's minimum, ``[batch, num_heads, query, key]``: the softmax's input, before
        the forward flattens it."""
        return value

    @EProperty(_pattern, description="The attention pattern the values are mixed with")
    def attention_probabilities(self, value) -> Pattern:
        return value if value.dim() == 4 else _heads_first(self, value)

    @attention_probabilities.postprocess
    def attention_probabilities(self, value):
        return _flat(value) if _dropping(self) else value

    @EProperty("source.attn_output_2.output", description=Attention.attention_head_outputs.description)
    def attention_head_outputs(self, value) -> HeadOutputs:
        """The ``bmm`` result viewed ``[batch, seq, num_heads, head_dim]``, before the heads are merged."""
        return value


class Mlp(Mlp):
    """XGLM has no MLP module; this class exists so the missing values are reported, never instantiated."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``. No MLP module to key.
ENVOYS = {XGLMDecoderLayer: Layer, XGLMAttention: Attention}


# -- sizes: what XGLM's config calls them -------------------------------------------

def intermediate_size(model: "StandardizedTransformer") -> int:
    """The width of the block's ``fc1``/``fc2`` path is ``ffn_dim``."""
    return model.config.ffn_dim
