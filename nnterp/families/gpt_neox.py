"""GPT-NeoX (``GPTNeoXForCausalLM``; Pythia, GPT-NeoX-20B).

Its tree is ``gpt_neox.{embed_in, emb_dropout, layers[i].{input_layernorm,
post_attention_layernorm, attention, mlp}, final_layer_norm}``. The head is
``lm_head`` on current transformers and was ``embed_out`` before; that key
binds only where it resolves, so both spellings map to ``lm_head``.
"""

from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention, GPTNeoXLayer, GPTNeoXMLP

from ..components import Attention, Layer, Mlp

RENAME = {
    "gpt_neox.embed_in": "embed_tokens",
    "gpt_neox.layers": "layers",
    "gpt_neox.final_layer_norm": "norm",
    "attention": "self_attn",
    "embed_out": "lm_head",
}


class Layer(Layer):
    """GPT-NeoX's decoder block; returns a bare tensor, so the base holds.

    With ``use_parallel_residual`` (Pythia's default) the block is parallel:
    ``x + attn(ln1(x)) + mlp(ln2(x))``. The contribution identity holds either
    way; only the norm names change meaning.
    """


class Attention(Attention):
    """GPT-NeoX's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """GPT-NeoX's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {GPTNeoXLayer: Layer, GPTNeoXAttention: Attention, GPTNeoXMLP: Mlp}
