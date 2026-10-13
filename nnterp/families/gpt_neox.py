"""GPT-NeoX (``GPTNeoXForCausalLM``; Pythia, GPT-NeoX-20B).

Its tree is ``gpt_neox.{embed_in, emb_dropout, layers[i].{input_layernorm,
post_attention_layernorm, attention, mlp}, final_layer_norm}``. The head is
``lm_head`` on current transformers and was ``embed_out`` before; that key
binds only where it resolves, so both spellings map to ``lm_head``.
"""

from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention, GPTNeoXLayer, GPTNeoXMLP

from ..components import Attention, EProperty, Layer, Mlp, PARALLEL, Residual

RENAME = {
    "gpt_neox.embed_in": "embed_tokens",
    "gpt_neox.layers": "layers",
    "gpt_neox.final_layer_norm": "norm",
    "attention": "self_attn",
    "embed_out": "lm_head",
    # The sublayers' interiors: the output projection, the activation and the down projection under their standard names.
    "dense": "o_proj",
    "act": "act_fn",
    "dense_4h_to_h": "down_proj",
}


class Layer(Layer):
    """GPT-NeoX's decoder block; returns a bare tensor, so the base holds.

    With ``use_parallel_residual`` (Pythia's default) the block is parallel:
    ``x + attn(ln1(x)) + mlp(ln2(x))``. The contribution identity holds either
    way; only the norm names change meaning.

    With ``use_parallel_residual`` there is no stream between the sublayers, and the MLP reads
    ``input_layernorm``'s output; without it the block is the pre-norm one.
    """

    layer_mid = EProperty(
        "post_attention_layernorm.input",
        description="The residual stream after the attention add",
        unavailable=lambda block: PARALLEL if block.root.config.use_parallel_residual else None,
    )

    @EProperty(
        lambda block: "input_layernorm.output" if block.root.config.use_parallel_residual else "post_attention_layernorm.output",
        description="The normed stream entering the MLP",
    )
    def mlp_norm_output(self, value) -> Residual:
        return value



class Attention(Attention):
    """GPT-NeoX's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """GPT-NeoX's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {GPTNeoXLayer: Layer, GPTNeoXAttention: Attention, GPTNeoXMLP: Mlp}
