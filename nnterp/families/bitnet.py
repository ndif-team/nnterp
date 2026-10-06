"""BitNet b1.58 (``BitNetForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward.
The attention norms its per-head outputs (``attn_sub_norm``) after the
interface and before ``o_proj``, so ``attention_head_outputs`` is before that
norm; the MLP norms its gated product (``ffn_sub_norm``) before ``down_proj``.
"""

from transformers.models.bitnet.modeling_bitnet import BitNetAttention, BitNetDecoderLayer, BitNetMLP

from ..components import Attention, Layer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """BitNet b1.58's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """BitNet b1.58's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """BitNet b1.58's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {BitNetDecoderLayer: Layer, BitNetAttention: Attention, BitNetMLP: Mlp}
