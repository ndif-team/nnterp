"""Phi-3 / 3.5 / 4 (``Phi3ForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward. Queries, keys and values are indexed out of one fused ``qkv_proj``.
"""

from transformers.models.phi3.modeling_phi3 import Phi3Attention, Phi3DecoderLayer, Phi3MLP

from ..components import Attention, Layer, Mlp, projection_weight

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    # The sublayers' interiors: the output projection, the activation and the down projection under their standard names.
    "activation_fn": "act_fn",
}


class Layer(Layer):
    """Phi-3's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Phi-3's attention; the shared eager forward and the residual added in the block, so the base holds."""

    def value_weight(self, kv_head):
        """``qkv_proj``'s rows are every query, every key, then every value."""
        weight = projection_weight(self, "qkv_proj")[self.num_heads * self.head_dim:]
        return weight.unflatten(0, (2, self.num_kv_heads, self.head_dim))[1, kv_head]


class Mlp(Mlp):
    """Phi-3's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Phi3DecoderLayer: Layer, Phi3Attention: Attention, Phi3MLP: Mlp}
