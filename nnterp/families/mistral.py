"""Mistral (``MistralForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward.
"""

from transformers.models.mistral.modeling_mistral import MistralAttention, MistralDecoderLayer, MistralMLP

from ..components import Attention, Layer, Mlp

MODEL_TYPES = ("mistral",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    # The same text model inside a multimodal wrapper, loaded with task="image-text-to-text":
    # the mistral3 wrapper of Mistral Small 3.1 and 3.2.
    "model.language_model.embed_tokens": "embed_tokens",
    "model.language_model.layers": "layers",
    "model.language_model.norm": "norm",
}


class Layer(Layer):
    """Mistral's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Mistral's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """Mistral's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {MistralDecoderLayer: Layer, MistralAttention: Attention, MistralMLP: Mlp}
