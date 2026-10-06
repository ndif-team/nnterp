"""Qwen2 / Qwen2.5 (``Qwen2ForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, attention through the shared eager forward.
"""

from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2DecoderLayer, Qwen2MLP

from ..components import Attention, Layer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    # The same text model inside a multimodal wrapper, loaded with task="image-text-to-text":
    # LLaVA-OneVision, llava-interleave, InternVL, FastVLM, GOT-OCR2.
    "model.language_model.embed_tokens": "embed_tokens",
    "model.language_model.layers": "layers",
    "model.language_model.norm": "norm",
}


class Layer(Layer):
    """Qwen2's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Qwen2's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """Qwen2's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Qwen2DecoderLayer: Layer, Qwen2Attention: Attention, Qwen2MLP: Mlp}
