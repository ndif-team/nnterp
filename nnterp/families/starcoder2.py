"""StarCoder2 (``Starcoder2ForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward.
The norms are ``nn.LayerNorm``; the MLP is ``c_fc``/``c_proj`` with no gate; the
attention and MLP outputs pass a residual dropout inside their module, the
identity in eval.
"""

from transformers.models.starcoder2.modeling_starcoder2 import (
    Starcoder2Attention,
    Starcoder2DecoderLayer,
    Starcoder2MLP,
)

from ..components import Attention, Layer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """StarCoder2's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """StarCoder2's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """StarCoder2's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Starcoder2DecoderLayer: Layer, Starcoder2Attention: Attention, Starcoder2MLP: Mlp}
