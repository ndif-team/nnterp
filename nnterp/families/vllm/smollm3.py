"""SmolLM3 on vLLM, through its transformers backend (``SmolLM3ForCausalLM`` -> ``TransformersForCausalLM``).

vLLM has no SmolLM3 of its own: it runs transformers' modules, so the tree
and the block are transformers' (see `nnterp.components.vllm.transformers_backend`),
with Llama's names, the stream ``[1, tokens, hidden]`` and the engine's
attention layer mounted as the attention's ``attn`` child. The block is
Llama's, pre-norm with the residual added in the block, so the
contributions are the modules' outputs. Every fourth block's attention
skips the rotary embedding (``no_rope_layers``); the queries and keys are
served as the attention layer receives them, rotated or not.
"""

from transformers.models.smollm3.modeling_smollm3 import SmolLM3Attention, SmolLM3DecoderLayer, SmolLM3MLP

from ...components.vllm.transformers_backend import Attention, Layer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """SmolLM3's decoder block: called with the stream and returning it, so the base holds."""


class Attention(Attention):
    """SmolLM3's attention; its output is what the block adds, so the base holds."""


class Mlp(Mlp):
    """SmolLM3's MLP; its output is what the block adds, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {SmolLM3DecoderLayer: Layer, SmolLM3Attention: Attention, SmolLM3MLP: Mlp}
