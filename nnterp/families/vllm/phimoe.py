"""Phi-3.5-MoE on vLLM (``vllm.model_executor.models.phimoe``).

vLLM registers the architecture as ``PhiMoEForCausalLM`` (the name the
Microsoft checkpoints carry); transformers' class is ``PhimoeForCausalLM``,
and a checkpoint saved by transformers says that, so the engine needs
``hf_overrides={"architectures": ["PhiMoEForCausalLM"]}`` to load it.

Llama's names. The block is not fused, whatever its signature says: it is
called ``forward(positions, hidden_states, residual)``, ignores the
``residual`` it is handed, adds both residuals itself, and returns
``(hidden_states, residual)`` with the *whole* stream first (the second is
the stream after the attention). The norms are ``nn.LayerNorm``. The block's
feed-forward is ``block_sparse_moe``, a mixture of experts, aliased to
``mlp``; its output is what the block adds. The routing (sparsemixer) is
inside vLLM's fused MoE kernel and has no values here.

vLLM's logits leave out the head's bias. The checkpoints have one
(``lm_head_bias``) and vLLM loads it into ``lm_head.bias``, but its logits
processor never adds it, so ``logits`` (what the engine samples from) and
`project_on_vocab` are transformers' logits less ``lm_head.bias``; everything
before the head is transformers'.
"""

from vllm.model_executor.models.phimoe import PhiMoE, PhiMoEAttention, PhiMoEDecoderLayer

from ...components.vllm import Attention, Layer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "block_sparse_moe": "mlp",
}


class Layer(Layer):
    """The block: called with the positions and the stream, adding its residuals itself, returning ``(stream, residual)``."""

    STREAM = 1
    returns_tuple = True


class Attention(Attention):
    """The attention; its output is what the block adds, so the base holds."""


class Mlp(Mlp):
    """The mixture of experts; its output is what the block adds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {PhiMoEDecoderLayer: Layer, PhiMoEAttention: Attention, PhiMoE: Mlp}
