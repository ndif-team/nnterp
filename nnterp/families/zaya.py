"""ZAYA (``ZayaForCausalLM``, Zyphra's ZAYA1).

Llama's containers over a block with **residual scaling modules**: after each
sublayer a ``ZayaResidualScaling`` merges the sublayer's output ``o`` into the
stream ``r`` with learned per-channel scales and biases::

    merge(o, r) = (o + hidden_states_bias) * hidden_states_scale + (r + residual_bias) * residual_scale

    h = post_attention_residual_scale(self_attn(input_layernorm(x)), x)
    out = post_mlp_residual_scale(mlp(post_attention_layernorm(h)), h)

What each sublayer adds is its scaled term, ``(o + hidden_states_bias) *
hidden_states_scale``, so ``attention_output`` and ``mlp_output`` are that
product: a computed copy of the module's output, inverted on assignment, with a
transform carrying an in-place edit back, as on Granite. The stream itself is
also rescaled, so the contribution identity is
``layer_output == (h + residual_bias) * residual_scale + mlp_output`` with
``h == (input + residual_bias) * residual_scale + attention_output`` (each merge's
own parameters, ``layers[i].post_*_residual_scale._module``, whose envoys the
block hands each sublayer as ``merge``), the plain one only
at their initial values (scales one, biases zero). The model also scales and
shifts the embedding module's output before the first block.

The attention runs the shared eager forward; its queries and keys come from a
compressed convolutional projection (``qkv_proj``, a causal convolution over q/k
and a delayed half of the values) and ``qk_norm``, so there are no ``q_proj`` /
``k_proj`` on the module. The block returns ``(hidden_states,
prev_router_hidden_states)``: the mixture's router carries a state from one block
to the next. The mixture (``ZayaSparseMoeBlock``) is on every block and returns
``(hidden_states, router_state)``; its experts are ``moe_intermediate_size`` wide,
which the root's ``intermediate_size`` reports (the config has no dense width).
``qk_norm`` L2-normalizes queries and keys and multiplies the keys by a learned
per-head ``temp``; at its initial value of zero every key is zero and the pattern
uniform.

The mixture is a `Moe`; its router, ``gate`` (aliased ``router``), mixes the
previous block's router state into its input and scores ``num_experts + 1``
classes with a small MLP (``router_mlp``), whose output is ``router_logits``: the
last column is **skip**. A slot that picks it runs no expert: its weight is 0 and
its index **0**, an alias of expert 0, so usage counts mask ``expert_weights == 0``.
"""

from typing import TYPE_CHECKING

import torch
from nnsight.intervention.envoy import Envoy
from transformers.models.zaya.modeling_zaya import ZayaAttention, ZayaDecoderLayer, ZayaSparseMoeBlock

from ..components import (
    Attention, EProperty, Layer, Moe, Residual, RouterLogits, TokenEProperty, first_tensor, rewrap,
)

if TYPE_CHECKING:
    from ..standardized import StandardizedTransformer

MODEL_TYPES = ("zaya",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "gate": "router",
}


def _scale(value: torch.Tensor, merge: torch.nn.Module) -> torch.Tensor:
    return (value + merge.hidden_states_bias) * merge.hidden_states_scale


def _unscale(value: torch.Tensor, merge: torch.nn.Module) -> torch.Tensor:
    return value / merge.hidden_states_scale - merge.hidden_states_bias


def _scaled_back(edited: torch.Tensor, raw, merge: torch.nn.Module):
    """The module output an edited scaled copy stands for, rewrapped; an unedited copy hands ``raw`` back untouched."""
    served = first_tensor(raw)
    if torch.equal(edited, _scale(served, merge)):
        return raw
    unscaled = _unscale(edited, merge)
    return (unscaled, *raw[1:]) if isinstance(raw, tuple) else unscaled


class Layer(Layer):
    """ZAYA's decoder block; returns ``(hidden_states, prev_router_hidden_states)``.

    Each sublayer's contribution is scaled by the merge that follows it, a sibling
    module, so the block hands the attention and the MLP their merge's envoy
    (``merge``) when it is built; the envoy outlives a weight swap, so its
    ``_module`` is the loaded one.
    """

    returns_tuple = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.self_attn.merge = self.post_attention_residual_scale
        self.mlp.merge = self.post_mlp_residual_scale


class Attention(Attention):
    """ZAYA's attention: the shared eager forward; the block adds its output shifted and scaled by ``post_attention_residual_scale``."""

    #: Set by the block: the envoy of the merge that follows this sublayer.
    merge: Envoy

    @EProperty(key="output", description="What the attention adds to the residual stream: its output plus the merge's hidden_states_bias, times its hidden_states_scale")
    def attention_output(self, value) -> Residual:
        return _scale(first_tensor(value), self.merge._module)

    @attention_output.postprocess
    def attention_output(self, value):
        return rewrap(self, _unscale(value, self.merge._module))

    @attention_output.transform
    def attention_output(self, value, raw):
        return _scaled_back(value, raw, self.merge._module)


class Mlp(Moe):
    """ZAYA's mixture of experts (``(hidden_states, router_state)``); the block adds its output shifted and scaled by ``post_mlp_residual_scale``.

    ``router_logits`` are ``router.router_mlp``'s output, ``num_experts + 1``
    columns (the last is skip), already ``[batch, seq, ...]``.
    """

    #: Set by the block: the envoy of the merge that follows this sublayer.
    merge: Envoy

    @EProperty(key="output", description="What the MLP adds to the residual stream: its output plus the merge's hidden_states_bias, times its hidden_states_scale")
    def mlp_output(self, value) -> Residual:
        return _scale(first_tensor(value), self.merge._module)

    @mlp_output.postprocess
    def mlp_output(self, value):
        return rewrap(self, _unscale(value, self.merge._module))

    @mlp_output.transform
    def mlp_output(self, value, raw):
        return _scaled_back(value, raw, self.merge._module)

    @TokenEProperty("router.router_mlp.output", description="The router's logits, one per expert and a last one for skip, before the scoring")
    def router_logits(self, value) -> RouterLogits:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {ZayaDecoderLayer: Layer, ZayaAttention: Attention, ZayaSparseMoeBlock: Mlp}


def intermediate_size(model: "StandardizedTransformer") -> int:
    """The MLP width is one expert's, ``moe_intermediate_size``: every block is a mixture and the config has no dense width."""
    return model.config.moe_intermediate_size
