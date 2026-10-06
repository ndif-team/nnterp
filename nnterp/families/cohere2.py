"""Cohere 2 (``Cohere2ForCausalLM``, Command-R7B / Command-A).

Cohere's tree and parallel block (``x + attn + mlp`` from one LayerNorm), with
sliding-window and full attention layers interleaved (``config.layer_types``).
Only the sliding layers apply the rotary; the full layers use no position
encoding. Both kinds make the same ``attention_interface`` call, so every
interior value resolves on every block, and on a full layer
``attention_queries`` / ``attention_keys`` are the unrotated projections.

The model multiplies the head's output by ``config.logit_scale``, as Cohere's
does, so the family takes Cohere's ``project_on_vocab``.
"""

from transformers.models.cohere2.modeling_cohere2 import Cohere2Attention, Cohere2DecoderLayer, Cohere2MLP

from ..components import Attention, Layer, Mlp
from .cohere import project_on_vocab  # noqa: F401  the same head: the same logit_scale

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    # The same text model inside a multimodal wrapper, loaded with task="image-text-to-text":
    # AyaVisionForConditionalGeneration, Cohere2VisionForConditionalGeneration.
    "model.language_model.embed_tokens": "embed_tokens",
    "model.language_model.layers": "layers",
    "model.language_model.norm": "norm",
}


class Layer(Layer):
    """Cohere-2's parallel block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Cohere-2's attention; the shared eager forward on sliding and full layers alike, so the base holds."""


class Mlp(Mlp):
    """Cohere-2's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Cohere2DecoderLayer: Layer, Cohere2Attention: Attention, Cohere2MLP: Mlp}
