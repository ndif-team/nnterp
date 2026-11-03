"""Persimmon (``PersimmonForCausalLM``).

Llama's tree with Persimmon's final norm: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, final_layernorm}`` and ``lm_head``; the
residual added in the block, attention through the shared eager forward. The norms
are ``nn.LayerNorm``. The attention projects queries, keys and values with one fused
``query_key_value``, norms the queries and keys per head (``q_layernorm``,
``k_layernorm``, under ``qk_layernorm``) and applies the rotary embedding to
``partial_rotary_factor`` of each head, all before the interface. The MLP is
``dense_h_to_4h``/``dense_4h_to_h`` with no gate; the block's own ``dropout`` after it
is the identity in eval.
"""

import torch
from transformers.models.persimmon.modeling_persimmon import PersimmonAttention, PersimmonDecoderLayer, PersimmonMLP

from ..components import Attention, EProperty, Keys, Layer, Mlp, Queries, rotary_reason

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.final_layernorm": "norm",
    # The sublayers' interiors: the output projection, the activation and the down projection under their standard names.
    "dense": "o_proj",
    "act": "act_fn",
    "dense_4h_to_h": "down_proj",
}


class Layer(Layer):
    """Persimmon's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Persimmon's attention; the shared eager forward and the residual added in the block, and a partial rotary sliced outside its apply function."""

    # The rotary call receives each head's leading ``rotary_ndims`` channels
    # only; the values before the rotary are the whole heads it slices them
    # from (the heads-first bindings, after the optional ``q_layernorm`` / ``k_layernorm``),
    # so they keep the queries' and keys' layout.

    @EProperty("source.query_states_1.output", description=Attention.attention_queries_prerope.description, unavailable=rotary_reason)
    def attention_queries_prerope(self, value) -> Queries:
        return value

    @EProperty("source.key_states_1.output", description=Attention.attention_keys_prerope.description, unavailable=rotary_reason)
    def attention_keys_prerope(self, value) -> Keys:
        return value

    def apply_rotary(self, x, embedding):
        """Persimmon rotates the leading ``rotary_ndims`` channels of each head and passes the rest, outside its apply function."""
        width = self._module.rotary_ndims
        return torch.cat((super().apply_rotary(x[..., :width], embedding), x[..., width:]), dim=-1)


class Mlp(Mlp):
    """Persimmon's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {PersimmonDecoderLayer: Layer, PersimmonAttention: Attention, PersimmonMLP: Mlp}
