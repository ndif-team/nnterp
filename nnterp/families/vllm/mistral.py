"""Mistral on vLLM (``vllm.model_executor.models.mistral``).

Llama's names and Llama's fused block. The block is vLLM's own subclass of
Llama's; its attention and MLP are Llama's classes or Mistral's subclasses of
them, as the checkpoint's config picks, so both are keyed.
"""

from vllm.model_executor.models.llama import LlamaAttention, LlamaMLP
from vllm.model_executor.models.mistral import MistralAttention, MistralDecoderLayer, MistralMLP

from ...components.vllm import Attention, FusedLayer, Mlp

MODEL_TYPES = ('mistral',)

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
ENVOYS = {MistralDecoderLayer: Layer, MistralAttention: Attention, LlamaAttention: Attention, MistralMLP: Mlp, LlamaMLP: Mlp}
