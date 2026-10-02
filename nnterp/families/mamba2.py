"""Mamba-2 (``Mamba2ForCausalLM``).

A pure state-space model: no attention and no MLP. The tree::

    backbone.embeddings
    backbone.layers[i]       Mamba2Block
        .norm                RMSNorm, the block's only norm
        .mixer               Mamba2Mixer, an SSD mixer
    backbone.norm_f
    lm_head

Each block is ``hidden + mixer(norm(hidden))`` (the residual in float32 when
``residual_in_fp32``), so the block's one contribution is the mixer's output
and ``layer.input + linear_attn.attention_output == layer_output``. The mixer
is `nnterp.StateSpace` under the standard name ``linear_attn`` (the recurrent
mixer's name on every hybrid). The block's
``norm`` keeps its name: a ``"norm"`` key would also match the mixer's own
gated ``norm``, and its output is ``linear_attn.input``. There is no
``self_attn`` and no ``mlp``, so `support` lists neither.

The sizes are the mixer's: ``num_heads`` and ``head_dim`` are the SSD heads
(``config.num_heads``, ``config.head_dim``), and ``intermediate_size`` is the
mixer's inner width, ``expand * hidden_size``, what ``in_proj`` expands to and
``out_proj`` reads from (the model has no MLP). With ``residual_in_fp32`` the
residual stream is float32 whatever the weights are, and the model casts it
to ``lm_head``'s dtype and its logits to float32, so `project_on_vocab` is the
family's own.
"""

import torch
from transformers.models.mamba2.modeling_mamba2 import Mamba2Block, Mamba2Mixer

from ..components import Layer, StateSpace

MODEL_TYPES = ("mamba2",)

RENAME = {
    "backbone.embeddings": "embed_tokens",
    "backbone.layers": "layers",
    "backbone.norm_f": "norm",
    "mixer": "linear_attn",
}


class Layer(Layer):
    """Mamba-2's block, one norm and one mixer; returns a bare tensor, so the base holds."""


class StateSpace(StateSpace):
    """Mamba-2's SSD mixer; transformers' pure-torch scan and the residual added in the block, so the base holds."""


def project_on_vocab(model, hidden: torch.Tensor) -> torch.Tensor:
    """The logit lens as the model makes its logits: the final norm, ``lm_head`` in its own dtype, then float32.

    The residual stream is float32 when ``residual_in_fp32`` (the final
    norm keeps it so), so the model casts it to ``lm_head``'s dtype and its
    logits back to float32.
    """
    normed = model.norm(hidden)
    return model.lm_head(normed.to(model.lm_head.weight.dtype)).float()


def num_heads(model) -> int:
    """The SSD heads, ``config.num_heads``."""
    return model.config.num_heads


def intermediate_size(model) -> int:
    """The mixer's inner width, ``expand * hidden_size``: the model has no MLP."""
    return int(model.config.expand * model.config.hidden_size)


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Mamba2Block: Layer, Mamba2Mixer: StateSpace}
