"""Qwen3.5 / 3.6 text (``Qwen3_5ForCausalLM``, model_type ``qwen3_5_text``).

Llama's tree with a hybrid block: three blocks in four carry a gated DeltaNet
mixer, ``linear_attn``, the fourth ordinary attention, ``self_attn``
(``config.layer_types``). Each block has one or the other, never both, so on
a linear block every ``self_attn`` value is reported missing and the linear
values live at ``layers[i].linear_attn`` (see `nnterp.LinearAttention`). Every
released checkpoint is a ``qwen3_5`` wrapper (``Qwen3_5ForConditionalGeneration``):
``task="text-generation"`` (the default) builds ``Qwen3_5ForCausalLM`` out of it, and
``task="image-text-to-text"`` the wrapper, whose text stack sits at
``model.language_model``; ``RENAME`` carries both spellings.
"""

from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, Qwen3_5DecoderLayer, Qwen3_5GatedDeltaNet, Qwen3_5MLP

from ..components import Attention, Layer, LinearAttention, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    # The same text model inside a multimodal wrapper, loaded with task="image-text-to-text":
    # Qwen3_5ForConditionalGeneration (every Qwen3.5 / 3.6 checkpoint).
    "model.language_model.embed_tokens": "embed_tokens",
    "model.language_model.layers": "layers",
    "model.language_model.norm": "norm",
}


class Layer(Layer):
    """Qwen3.5's block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Qwen3.5's softmax attention (one block in four); the shared eager forward, so the base holds."""


class LinearAttention(LinearAttention):
    """Qwen3.5's gated DeltaNet mixer; transformers' pure-torch chunked rule, so the base holds."""


class Mlp(Mlp):
    """Qwen3.5's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Qwen3_5DecoderLayer: Layer, Qwen3_5Attention: Attention, Qwen3_5GatedDeltaNet: LinearAttention, Qwen3_5MLP: Mlp}
