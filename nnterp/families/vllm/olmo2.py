"""OLMo 2 on vLLM, through its transformers backend (``Olmo2ForCausalLM`` -> ``TransformersForCausalLM``).

vLLM has no OLMo 2 of its own: it runs transformers' modules, so the tree
and the block are transformers' (see `nnterp.components.vllm.transformers_backend`),
with Llama's names, the stream ``[1, tokens, hidden]`` and the engine's
attention layer mounted as the attention's ``attn`` child. The block is
post-norm, ``x + post_attention_layernorm(attn(x))`` and then ``+
post_feedforward_layernorm(mlp(x))``, so the contributions point at the
sibling norms, as on transformers. The attention norms its queries and keys
over every head at once before splitting them.
"""

from transformers.models.olmo2.modeling_olmo2 import Olmo2Attention, Olmo2DecoderLayer, Olmo2MLP

from ...components import Residual
from ...components.vllm import Flat
from ...components.vllm.transformers_backend import Attention, Layer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """OLMo-2's decoder block: called with the stream and returning it, so the base holds."""


class Attention(Attention):
    """OLMo-2's attention: what reaches the residual stream is the post-attention norm's output."""

    @Flat(
        "../post_attention_layernorm.output",
        batch=True,
        description="What the attention adds to the residual stream: the post-attention norm's output, [1, tokens, hidden]",
    )
    def attention_output(self, value) -> Residual:
        return value


class Mlp(Mlp):
    """OLMo-2's MLP: what reaches the residual stream is the post-feedforward norm's output."""

    @Flat(
        "../post_feedforward_layernorm.output",
        batch=True,
        description="What the MLP adds to the residual stream: the post-feedforward norm's output, [1, tokens, hidden]",
    )
    def mlp_output(self, value) -> Residual:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Olmo2DecoderLayer: Layer, Olmo2Attention: Attention, Olmo2MLP: Mlp}
