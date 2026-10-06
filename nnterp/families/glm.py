"""GLM-4 (the first generation: GLM-4-9B, ``GlmForCausalLM``) (``GlmForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward.
The rotary embedding covers ``partial_rotary_factor`` of each head, before the
interface. The MLP fuses its gate and up projections into one ``gate_up_proj``
and returns a bare tensor.
"""

from transformers.models.glm.modeling_glm import GlmAttention, GlmDecoderLayer, GlmMLP

from ..components import Attention, Layer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """GLM-4's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """GLM-4's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """GLM-4's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {GlmDecoderLayer: Layer, GlmAttention: Attention, GlmMLP: Mlp}
