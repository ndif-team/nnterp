"""Gemma 3 (text) on vLLM (``vllm.model_executor.models.gemma3``).

Llama's names and a fused block, with Gemma's sandwich norms: what the block
adds is each sublayer's *post* norm's output, not the module's, as on
transformers, so the contributions point at the sibling norms. The attention
norms its queries and keys. The ``sqrt(hidden_size)`` scaling of the
embeddings is applied by the model after ``embed_tokens``, not inside it as
on transformers, so ``token_embeddings`` is the unscaled lookup and
``layers[0].layer_input`` the scaled stream.
"""

from vllm.model_executor.models.gemma3 import Gemma3Attention, Gemma3DecoderLayer, Gemma3MLP

from ...components import Residual
from ...components.vllm import Attention, Flat, FusedLayer, Mlp

MODEL_TYPES = ("gemma3_text",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(FusedLayer):
    """Gemma-3's decoder block: ``forward(positions, hidden_states, residual) -> (hidden_states, residual)``."""


class Attention(Attention):
    """Gemma-3's attention: what reaches the residual stream is the post-attention norm's output."""

    @Flat(
        "../post_attention_layernorm.output",
        description="What the attention adds to the residual stream: the post-attention norm's output, [1, tokens, hidden]",
    )
    def attention_output(self, value) -> Residual:
        return value


class Mlp(Mlp):
    """Gemma-3's MLP: what reaches the residual stream is the post-feedforward norm's output."""

    @Flat(
        "../post_feedforward_layernorm.output",
        description="What the MLP adds to the residual stream: the post-feedforward norm's output, [1, tokens, hidden]",
    )
    def mlp_output(self, value) -> Residual:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Gemma3DecoderLayer: Layer, Gemma3Attention: Attention, Gemma3MLP: Mlp}
