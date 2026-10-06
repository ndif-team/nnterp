"""StableLM / StableLM-2 (``StableLmForCausalLM``).

Llama's containers. With ``use_parallel_residual`` (StableLM-2) the block is
parallel with one ``input_layernorm`` and a ``dropout``; without it there is a
``post_attention_layernorm`` too. The attention norms its queries and keys
(``q_layernorm`` / ``k_layernorm``).
"""

from transformers.models.stablelm.modeling_stablelm import StableLmAttention, StableLmDecoderLayer, StableLmMLP

from ..components import Attention, Layer, Mlp

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """StableLM's block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """StableLM's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """StableLM's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {StableLmDecoderLayer: Layer, StableLmAttention: Attention, StableLmMLP: Mlp}
