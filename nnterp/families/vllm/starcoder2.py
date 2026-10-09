"""StarCoder2 on vLLM, through its transformers backend (``Starcoder2ForCausalLM`` -> ``TransformersForCausalLM``).

vLLM has no StarCoder2 of its own: it runs transformers' modules, so the
tree and the block are transformers' (see
`nnterp.components.vllm.transformers_backend`), with Llama's names, the
stream ``[1, tokens, hidden]`` and the engine's attention layer mounted as
the attention's ``attn`` child. The block is Llama's, pre-norm with the
residual added in the block, so the contributions are the modules' outputs.
The norms are layer norms and the MLP is ``c_fc``/``c_proj`` with no gate.
"""

from transformers.models.starcoder2.modeling_starcoder2 import (
    Starcoder2Attention,
    Starcoder2DecoderLayer,
    Starcoder2MLP,
)

from ...components.vllm.transformers_backend import Attention, Layer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """StarCoder2's decoder block: called with the stream and returning it, so the base holds."""


class Attention(Attention):
    """StarCoder2's attention; its output is what the block adds, so the base holds."""


class Mlp(Mlp):
    """StarCoder2's MLP; its output is what the block adds, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Starcoder2DecoderLayer: Layer, Starcoder2Attention: Attention, Starcoder2MLP: Mlp}
