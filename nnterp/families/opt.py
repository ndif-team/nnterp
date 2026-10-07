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

from ..components import Attention, Layer, Mlp

if TYPE_CHECKING:
    from ..standardized import StandardizedTransformer

RENAME = {
    "model.decoder.embed_tokens": "embed_tokens",
    "model.decoder.layers": "layers",
    "model.decoder.final_layer_norm": "norm",
    "self_attn_layer_norm": "input_layernorm",
}


class Layer(Layer):
    """OPT's decoder block; returns a bare tensor, so the base holds."""


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
