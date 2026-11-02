"""ERNIE 4.5 dense on vLLM (``vllm.model_executor.models.ernie45``).

vLLM runs ERNIE 4.5 as a Llama: ``Ernie4_5ForCausalLM`` subclasses its
``LlamaForCausalLM``, builds Llama's own modules and then switches each
attention's rotary embedding to the interleaved (not NeoX) layout and drops
the output projection's bias, which is what transformers' module computes.
So the names, the fused block and the classes are Llama's.
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
