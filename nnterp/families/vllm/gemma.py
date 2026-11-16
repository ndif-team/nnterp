"""Gemma on vLLM (``vllm.model_executor.models.gemma``).

Llama's names and Llama's fused block. One thing is not transformers': the
``sqrt(hidden_size)`` scaling of the embeddings is applied by the model after
``embed_tokens``, not inside it as on transformers, so ``token_embeddings``
is the unscaled lookup and ``layers[0].layer_input`` the scaled stream.
"""

from vllm.model_executor.models.gemma import GemmaAttention, GemmaDecoderLayer, GemmaMLP

from ...components.vllm import Attention, FusedLayer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(FusedLayer):
    """The decoder block: ``forward(positions, hidden_states, residual) -> (hidden_states, residual)``."""


class Attention(Attention):
    """The attention; its output is what the block adds, so the base holds."""


class Mlp(Mlp):
    """The MLP; its output is what the block adds, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {GemmaDecoderLayer: Layer, GemmaAttention: Attention, GemmaMLP: Mlp}
