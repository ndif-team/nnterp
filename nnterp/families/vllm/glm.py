"""GLM-4, the first generation (``GlmForCausalLM``), on vLLM (``vllm.model_executor.models.glm``).

vLLM runs it as a Llama: ``GlmForCausalLM`` subclasses its
``LlamaForCausalLM``, builds Llama's own modules and then switches each
attention's rotary embedding to the interleaved (not NeoX) layout and drops
the output projection's bias. So the names, the fused block and the classes
are Llama's; the MLP's gate and up projections are fused into one
``gate_up_proj`` on both engines.
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
