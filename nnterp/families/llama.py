"""Llama (``LlamaForCausalLM``), the family the standard vocabulary is taken from.

Block names (``input_layernorm``, ``self_attn``, ``post_attention_layernorm``,
``mlp``) and ``lm_head`` are already the standard ones. The only change is
lifting the containers out of ``model.model``: ``model.layers`` instead of
``model.model.layers``. Multimodal wrappers around a Llama text model keep it at
``model.language_model`` (Llava, DeepSeek-VL, Janus) or ``model.text_model``
(Idefics 3, SmolVLM); ``RENAME`` carries those spellings too.
"""

from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer, LlamaMLP

from ..components import Attention, Layer, Mlp

MODEL_TYPES = ("llama",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    # The same text model inside a multimodal wrapper, loaded with task="image-text-to-text":
    # Llava 1.5, VipLlava, LLaVA-NeXT, DeepSeek-VL, Janus ...
    "model.language_model.embed_tokens": "embed_tokens",
    "model.language_model.layers": "layers",
    "model.language_model.norm": "norm",
    # ... and Idefics 3 / SmolVLM.
    "model.text_model.embed_tokens": "embed_tokens",
    "model.text_model.layers": "layers",
    "model.text_model.norm": "norm",
}


class Layer(Layer):
    """Llama's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Llama's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """Llama's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {LlamaDecoderLayer: Layer, LlamaAttention: Attention, LlamaMLP: Mlp}
