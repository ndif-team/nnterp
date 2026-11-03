"""Phi 1 / 1.5 / 2 (``PhiForCausalLM``).

Llama's containers with a parallel block: one ``input_layernorm`` feeds both
sublayers and ``x + attn + mlp`` is summed at the end (``resid_dropout`` on
each, inert in eval). The final norm is ``final_layernorm``. There is no
``post_attention_layernorm``.
"""

import torch
from transformers.models.phi.modeling_phi import PhiAttention, PhiDecoderLayer, PhiMLP

from ..components import Attention, EProperty, Keys, Layer, Mlp, PARALLEL, Queries, Residual, rotary_reason, unavailable


def _rotated_from(name):
    """The binding the rotary's slice is taken from: the projection's, or the q/k layernorm's where the checkpoint has one."""

    def path(attn):
        return f"source.{name}_{1 if attn._module.qk_layernorm else 0}.output"

    return path


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
    """Phi's attention; the shared eager forward and the residual added in the block, and a partial rotary sliced outside its apply function."""

    # The rotary call receives each head's leading ``rotary_ndims`` channels
    # only; the values before the rotary are the whole heads it slices them
    # from (the bindings after the optional ``q_layernorm`` / ``k_layernorm``),
    # so they keep the queries' and keys' layout.

    @EProperty(_rotated_from("query_states"), description=Attention.attention_queries_prerope.description, unavailable=rotary_reason)
    def attention_queries_prerope(self, value) -> Queries:
        return value

    @EProperty(_rotated_from("key_states"), description=Attention.attention_keys_prerope.description, unavailable=rotary_reason)
    def attention_keys_prerope(self, value) -> Keys:
        return value

    def apply_rotary(self, x, embedding):
        """Phi rotates the leading ``rotary_ndims`` channels of each head and passes the rest, outside its apply function."""
        width = self._module.rotary_ndims
        return torch.cat((super().apply_rotary(x[..., :width], embedding), x[..., width:]), dim=-1)


class Mlp(Mlp):
    """Phi's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {PhiDecoderLayer: Layer, PhiAttention: Attention, PhiMLP: Mlp}
