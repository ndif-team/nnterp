"""Phi-3 on vLLM (``vllm.model_executor.models.phi3``).

vLLM runs Phi-3 as a Llama: ``Phi3ForCausalLM`` subclasses its
``LlamaForCausalLM`` and builds Llama's own modules, so the names, the fused
block and the classes are Llama's.
"""

from vllm.model_executor.models.llama import LlamaAttention, LlamaDecoderLayer, LlamaMLP

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
ENVOYS = {LlamaDecoderLayer: Layer, LlamaAttention: Attention, LlamaMLP: Mlp}
