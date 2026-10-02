"""Falcon-Mamba (``FalconMambaForCausalLM``): Mamba-1 with normed scan inputs.

Mamba's tree and block: ``backbone.{embeddings, layers[i].{norm, mixer},
norm_f}`` plus ``lm_head``, a pre-norm and a selective-scan mixer per block,
no attention and no MLP (see `nnterp.families.mamba`). What differs is inside
the mixer: weightless RMS norms (``dt_layernorm``, ``b_layernorm``,
``c_layernorm``) on the step-size projection, ``B`` and ``C`` before the
scan. The values are read at the kernel call, after those norms, so
``attention_keys`` and ``attention_queries`` are the normed ``B`` and ``C``
the scan uses, and the base holds. The kernels are the same functions under
the same names, in Falcon-Mamba's own modeling module:
``nnterp.route_kernels(model.family, "torch")`` routes them.
"""

from typing import TYPE_CHECKING

import torch
from transformers.models.falcon_mamba.modeling_falcon_mamba import FalconMambaBlock, FalconMambaMixer

from ..components import Layer, SelectiveScan

if TYPE_CHECKING:
    from ..standardized import StandardizedTransformer

MODEL_TYPES = ("falcon_mamba",)

RENAME = {
    "backbone.embeddings": "embed_tokens",
    "backbone.layers": "layers",
    "backbone.norm_f": "norm",
    "mixer": "linear_attn",
    "norm": "input_layernorm",
}


class Layer(Layer):
    """Falcon-Mamba's block; returns a bare tensor, so the base holds."""


class SelectiveScan(SelectiveScan):
    """Falcon-Mamba's mixer; the norms on ``B``, ``C`` and the step size come before the scan, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {FalconMambaBlock: Layer, FalconMambaMixer: SelectiveScan}


def project_on_vocab(model: "StandardizedTransformer", hidden: torch.Tensor) -> torch.Tensor:
    """The logit lens as the model makes its logits: the final norm, ``lm_head`` in its own dtype, then float32.

    With ``residual_in_fp32`` the stream is float32 whatever the load dtype, so
    the head's input is cast to the head's dtype, and the logits come back
    float32, as the model's own do.
    """
    return model.lm_head(model.norm(hidden).to(model.lm_head.weight.dtype)).float()
