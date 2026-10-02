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

from transformers.models.persimmon.modeling_persimmon import PersimmonAttention, PersimmonDecoderLayer, PersimmonMLP

from ..components import Attention, Layer, Mlp

MODEL_TYPES = ("persimmon",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.final_layernorm": "norm",
}


class Layer(Layer):
    """Persimmon's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Persimmon's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """Persimmon's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {PersimmonDecoderLayer: Layer, PersimmonAttention: Attention, PersimmonMLP: Mlp}
