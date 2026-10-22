"""OPT (``OPTForCausalLM``).

``model.decoder.{embed_tokens, embed_positions, layers[i].{self_attn,
self_attn_layer_norm, fc1, activation_fn, fc2, final_layer_norm},
final_layer_norm}`` and ``lm_head``. There is no MLP module: ``fc1`` and
``fc2`` sit on the block, so ``mlp`` and ``mlp_output`` do not exist here
and `StandardizedTransformer.support` says so. The block's own
``final_layer_norm`` is the pre-MLP norm; it keeps its native name, since a
single-component alias would also bind on the decoder's final norm.

The MLP path runs on the flattened ``[batch * seq, ...]`` stream, so
``input + attention_output + fc2.output.view_as(layer_output) == layer_output``.
The queries are scaled by ``head_dim**-0.5`` at ``q_proj``'s output, so
``attention_queries`` are pre-scaled. ``facebook/opt-350m`` sets
``do_layer_norm_before`` false (norms after the adds, so that identity does not
hold), has no final norm (no ``norm``; ``project_on_vocab`` raises
``AttributeError``) and wraps the stack in ``project_in`` / ``project_out``
(512 <-> 1024): ``token_embeddings`` is 512 wide and ``layers[0].input`` 1024.
"""

from typing import TYPE_CHECKING

from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer

from ..components import POST_NORM_ATTENTION, POST_NORM_MLP, Attention, EProperty, FlatResidual, Layer, Mlp

if TYPE_CHECKING:
    from ..standardized import StandardizedTransformer

RENAME = {
    "model.decoder.embed_tokens": "embed_tokens",
    "model.decoder.layers": "layers",
    "model.decoder.final_layer_norm": "norm",
    "self_attn_layer_norm": "input_layernorm",
    # The sublayers' interiors: the output projection, the activation and the down projection under their standard names.
    "out_proj": "o_proj",
}


def norms_before(block) -> bool:
    """``do_layer_norm_before``: the norms precede the sublayers (every OPT but the 350m, whose norms follow the adds)."""
    return block.root.config.do_layer_norm_before


class Layer(Layer):
    """OPT's decoder block; returns a bare tensor, so the base holds.

    The pre-MLP norm is the block's own ``final_layer_norm``, so ``layer_mid`` and ``mlp_norm_output``
    read there, flattened to ``[batch * seq, hidden]`` as the block holds the stream around its feed-forward
    path (``view(-1, hidden)``, and back at the end); there is no ``mlp`` module, so ``mlp_input`` does not exist.
    With ``do_layer_norm_before`` false (OPT-350m) the norms follow the adds: the attention reads the stream
    itself, ``layer_mid`` is ``self_attn_layer_norm``'s output, and no norm precedes the feed-forward path.
    """

    attention_norm_output = EProperty(
        "input_layernorm.output", description="The normed stream entering the attention",
        unavailable=lambda block: None if norms_before(block) else POST_NORM_ATTENTION,
    )

    @EProperty(
        lambda block: "final_layer_norm.input" if norms_before(block) else "input_layernorm.output",
        description="The residual stream after the attention add, flattened to [batch * seq, hidden]",
    )
    def layer_mid(self, value) -> FlatResidual:
        return value.reshape(-1, value.shape[-1])  # a view: the block's own flat rows, or the norm's [batch, seq, hidden] output

    @layer_mid.postprocess
    def layer_mid(self, value):
        current = self.final_layer_norm.input if norms_before(self) else self.input_layernorm.output
        return value.view(current.shape)

    mlp_norm_output = EProperty(
        "final_layer_norm.output", description="The normed stream entering the feed-forward path, flattened to [batch * seq, hidden]",
        unavailable=lambda block: None if norms_before(block) else POST_NORM_MLP,
    )



class Attention(Attention):
    """OPT's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """OPT has no MLP module; this class exists so the missing values are reported, never instantiated."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``. No MLP module to key.
ENVOYS = {OPTDecoderLayer: Layer, OPTAttention: Attention}


# -- sizes: what OPT's config calls them --------------------------------------------

def intermediate_size(model: "StandardizedTransformer") -> int:
    """The width of the block's ``fc1``/``fc2`` path is ``ffn_dim``."""
    return model.config.ffn_dim
