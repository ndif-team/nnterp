"""Mamba (``MambaForCausalLM``): Mamba-1, a pure state-space model.

Its tree is ``backbone.{embeddings, layers[i].{norm, mixer}, norm_f}`` plus
``lm_head``. A block is a pre-norm and a selective-scan mixer, nothing else:
``layer_output = input + mixer(norm(input))``, so there is no ``self_attn``
and no ``mlp``, and the identity is ``input + linear_attn.attention_output
== layer_output``. The mixer answers to ``linear_attn``, the standard name
of a recurrent mixer, and carries the `SelectiveScan` values; the block's
norm answers to ``input_layernorm``.

The kernels: with ``mamba_ssm`` installed the forward dispatches the scan
and the decode step to its compiled CUDA kernels, which have no source to
read inside and do not run on CPU; ``nnterp.route_kernels(model.family,
"torch")`` before the first trace binds them to transformers' pure-torch
functions. With ``residual_in_fp32`` the block adds in float32, so
``layer_output`` is float32 whatever the load dtype; the model casts the
stream to ``lm_head``'s dtype and returns float32 logits, and so does the
family's `project_on_vocab`. The config's
``intermediate_size`` is the mixer's inner width (``expand * hidden_size``,
the channels of the state); there are no attention heads, so the head sizes
have nothing to read.
"""

from typing import TYPE_CHECKING

import torch
from transformers.models.mamba.modeling_mamba import MambaBlock, MambaMixer

from ..components import Layer, SelectiveScan

if TYPE_CHECKING:
    from ..standardized import StandardizedTransformer

RENAME = {
    "backbone.embeddings": "embed_tokens",
    "backbone.layers": "layers",
    "backbone.norm_f": "norm",
    "mixer": "linear_attn",
    "norm": "input_layernorm",
}


class Layer(Layer):
    """Mamba's block; returns a bare tensor, so the base holds."""


class SelectiveScan(SelectiveScan):
    """Mamba's mixer; transformers' pure-torch selective scan and state update, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {MambaBlock: Layer, MambaMixer: SelectiveScan}


def project_on_vocab(model: "StandardizedTransformer", hidden: torch.Tensor) -> torch.Tensor:
    """The logit lens as the model makes its logits: the final norm, ``lm_head`` in its own dtype, then float32.

    With ``residual_in_fp32`` the stream is float32 whatever the load dtype, so
    the head's input is cast to the head's dtype, and the logits come back
    float32, as the model's own do.
    """
    return model.lm_head(model.norm(hidden).to(model.lm_head.weight.dtype)).float()
