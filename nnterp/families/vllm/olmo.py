"""OLMo 1 on vLLM, through its transformers backend (``OlmoForCausalLM`` -> ``TransformersForCausalLM``).

vLLM has no OLMo of its own: it runs transformers' modules, so the tree and
the block are transformers' (see `nnterp.components.vllm.transformers_backend`),
with Llama's names, the stream ``[1, tokens, hidden]`` and the engine's
attention layer mounted as the attention's ``attn`` child. The block is
Llama's, pre-norm with the residual added in the block, so the
contributions are the modules' outputs. The norms are OLMo's non-parametric
layer norms.
"""

from transformers.models.olmo.modeling_olmo import OlmoAttention, OlmoDecoderLayer, OlmoMLP

from ...components.vllm.transformers_backend import Attention, Layer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """OLMo's decoder block: called with the stream and returning it, so the base holds."""


class Attention(Attention):
    """OLMo's attention; its output is what the block adds, so the base holds."""


class Mlp(Mlp):
    """OLMo's MLP; its output is what the block adds, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {OlmoDecoderLayer: Layer, OlmoAttention: Attention, OlmoMLP: Mlp}
