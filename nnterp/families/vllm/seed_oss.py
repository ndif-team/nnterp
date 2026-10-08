"""Seed-OSS on vLLM (``vllm.model_executor.models.seed_oss``).

Llama's names and Llama's fused block, with vLLM's own classes: the block
takes and returns the stream as ``(hidden_states, residual)``. On a
checkpoint that ties its embeddings vLLM's ``lm_head`` is the embedding
module itself, which still unembeds through the logits processor.
"""

from vllm.model_executor.models.seed_oss import SeedOssAttention, SeedOssDecoderLayer, SeedOssMLP

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
ENVOYS = {SeedOssDecoderLayer: Layer, SeedOssAttention: Attention, SeedOssMLP: Mlp}
