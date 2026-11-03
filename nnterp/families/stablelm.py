"""StableLM / StableLM-2 (``StableLmForCausalLM``).

Llama's containers. With ``use_parallel_residual`` the block is parallel with one
``input_layernorm`` and a ``dropout``; without it there is a
``post_attention_layernorm`` too. Under ``qk_layernorm`` the attention norms its
queries and keys per head (``q_layernorm`` / ``k_layernorm``). Of the released
checkpoints only StableLM-2-12B sets both (as does the pinned tiny); StableLM-2-1.6B
and StableLM-3B-4E1T are sequential with no q/k norms.
"""

import torch
from transformers.models.stablelm.modeling_stablelm import StableLmAttention, StableLmDecoderLayer, StableLmMLP

from ..components import Attention, EProperty, Keys, Layer, Mlp, PARALLEL, Queries, Residual, rotary_reason


def _rotated_from(name):
    """The binding the rotary's slice is taken from: the heads-first view's, or the q/k layernorm's where the checkpoint has one."""

    def path(attn):
        return f"source.{name}_{2 if attn._module.qk_layernorm else 1}.output"

    return path


RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """StableLM's block; returns a bare tensor, so the base holds.

    With ``use_parallel_residual`` there is no stream between the sublayers, and the MLP reads
    ``input_layernorm``'s output; without it the block is the pre-norm one.
    """

    @EProperty(
        "post_attention_layernorm.input",
        description="The residual stream after the attention add",
        unavailable=lambda block: PARALLEL if block.root.config.use_parallel_residual else None,
    )
    def layer_mid(self, value) -> Residual:
        return value

    @EProperty(
        lambda block: "input_layernorm.output" if block.root.config.use_parallel_residual else "post_attention_layernorm.output",
        description="The normed stream entering the MLP",
    )
    def mlp_norm_output(self, value) -> Residual:
        return value



class Attention(Attention):
    """StableLM's attention; the shared eager forward and the residual added in the block, and a partial rotary sliced outside its apply function."""

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
        """StableLM rotates the leading ``rotary_ndims`` channels of each head and passes the rest, outside its apply function."""
        width = self._module.rotary_ndims
        return torch.cat((super().apply_rotary(x[..., :width], embedding), x[..., width:]), dim=-1)


class Mlp(Mlp):
    """StableLM's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {StableLmDecoderLayer: Layer, StableLmAttention: Attention, StableLmMLP: Mlp}
