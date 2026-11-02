"""StableLM / StableLM-2 on vLLM (``vllm.model_executor.models.stablelm``).

Llama's names, but the block is not fused, whatever its return type says: it
is called ``forward(positions, hidden_states)`` with the stream and returns
``(hidden_states, residual)`` with the *whole* stream first; the second element
is the stale stream after the attention, which the model drops, so summing
the pair is wrong. Each sublayer's output is what the block adds, as on
transformers.

vLLM implements the sequential block only: it reads neither
``use_parallel_residual`` nor ``qk_layernorm``, so on a checkpoint that sets
them (StableLM-2-12B) vLLM's forward is not the model's. StableLM-2-1.6B and
StableLM-3B-4E1T set neither.
"""

from vllm.model_executor.models.stablelm import StablelmAttention, StablelmDecoderLayer, StablelmMLP

from ...components.vllm import Attention, Layer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """The block: called with the positions and the stream, returning ``(stream, stale residual)``."""

    STREAM = 1
    returns_tuple = True


class Attention(Attention):
    """The attention; its output is what the block adds, so the base holds."""


class Mlp(Mlp):
    """The MLP; its output is what the block adds, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {StablelmDecoderLayer: Layer, StablelmAttention: Attention, StablelmMLP: Mlp}
