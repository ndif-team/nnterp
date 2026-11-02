"""Phi 1 / 1.5 / 2 (``PhiForCausalLM``).

Llama's containers with a parallel block: one ``input_layernorm`` feeds both
sublayers and ``x + attn + mlp`` is summed at the end (``resid_dropout`` on
each, inert in eval). The final norm is ``final_layernorm``. There is no
``post_attention_layernorm``.
"""

from transformers.models.phi.modeling_phi import PhiAttention, PhiDecoderLayer, PhiMLP

from ..components import Attention, EProperty, Layer, Mlp, PARALLEL, Residual, unavailable

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.final_layernorm": "norm",
    # The sublayers' interiors: the output projection, the activation and the down projection under their standard names.
    "dense": "o_proj",
    "activation_fn": "act_fn",
    "fc2": "down_proj",
}


class Layer(Layer):
    """Phi's parallel block; returns a bare tensor, so the base holds.

    A parallel block: there is no stream between the sublayers, and the MLP reads the same normed input
    as the attention.
    """

    layer_mid = unavailable(PARALLEL)

    @EProperty("input_layernorm.output", description="The normed stream entering the MLP: the one norm feeds both sublayers")
    def mlp_norm_output(self, value) -> Residual:
        return value



class Attention(Attention):
    """Phi's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """Phi's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {PhiDecoderLayer: Layer, PhiAttention: Attention, PhiMLP: Mlp}
