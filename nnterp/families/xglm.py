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
``[batch * heads, seq, head_dim]`` for ``torch.bmm``, the mask added and clamped,
softmax, dropout, ``bmm`` with the values; the interior needs no eager load.
"""

from typing import TYPE_CHECKING

import torch
from transformers.models.xglm.modeling_xglm import XGLMAttention, XGLMDecoderLayer

from ..components import Attention, EProperty, HeadOutputs, Keys, Layer, Mlp, Pattern, Queries, Values, seq_first

if TYPE_CHECKING:
    from nnsight import Envoy

    from ..standardized import StandardizedTransformer

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.layer_norm": "norm",
    "self_attn_layer_norm": "input_layernorm",
}


def _softmax(envoy: "Envoy") -> str:
    """The softmax the forward runs: an fp32 one cast back on an fp16 model, the plain one otherwise."""
    fp16 = envoy._module.q_proj.weight.dtype == torch.float16
    return f"source.nn_functional_softmax_{0 if fp16 else 1}.input"


_softmax.__name__ = "nn_functional_softmax_0|nn_functional_softmax_1"


def _heads_first(envoy: "Envoy", value: torch.Tensor) -> torch.Tensor:
    """A ``[batch * heads, ...]`` tensor viewed as ``[batch, heads, ...]``."""
    heads = envoy._module.num_heads
    return value.view(value.shape[0] // heads, heads, *value.shape[1:])


def _flat(value: torch.Tensor) -> torch.Tensor:
    """A ``[batch, heads, ...]`` tensor back to the ``[batch * heads, ...]`` the forward holds."""
    return value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])


class Layer(Layer):
    """XGLM's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """XGLM's attention: the residual is added in the block; the interior on its own ``bmm`` arithmetic."""

    # The queries, keys and values are the three ``reshape`` calls that flatten
    # them for ``bmm`` (the keys and values after the cache update), viewed back
    # to heads first; the scores are the softmax's input, masked; the pattern is
    # the dropout's output; the head outputs are the ``bmm`` result viewed back to
    # ``[batch, heads, seq, head_dim]``, served tokens first. Every read is a view
    # of the tensor the forward holds, so in-place edits reach the model, and a
    # write is reshaped back to the forward's layout.

    @EProperty(
        "source.query_states_reshape_0.output",
        description="The queries entering attention, already scaled by 1/sqrt(head_dim)",
    )
    def attention_queries(self, value) -> Queries:
        return _heads_first(self, value)

    @attention_queries.postprocess
    def attention_queries(self, value):
        return _flat(value)

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

    @EProperty(_softmax, description=Attention.attention_scores.description)
    def attention_scores(self, value) -> Pattern:
        return _heads_first(self, value)

    @attention_scores.postprocess
    def attention_scores(self, value):
        return _flat(value)

    @EProperty(
        "source.nn_functional_dropout_0.output",
        description="The attention pattern the values are mixed with",
    )
    def attention_probabilities(self, value) -> Pattern:
        return _heads_first(self, value)

    @attention_probabilities.postprocess
    def attention_probabilities(self, value):
        return _flat(value)

    @EProperty("source.torch_bmm_1.output", description=Attention.attention_head_outputs.description)
    def attention_head_outputs(self, value) -> HeadOutputs:
        return seq_first(_heads_first(self, value))

    @attention_head_outputs.postprocess
    def attention_head_outputs(self, value):
        return _flat(seq_first(value))


class Mlp(Mlp):
    """XGLM has no MLP module; this class exists so the missing values are reported, never instantiated."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``. No MLP module to key.
ENVOYS = {XGLMDecoderLayer: Layer, XGLMAttention: Attention}


# -- sizes: what XGLM's config calls them -------------------------------------------

def intermediate_size(model: "StandardizedTransformer") -> int:
    """The width of the block's ``fc1``/``fc2`` path is ``ffn_dim``."""
    return model.config.ffn_dim
