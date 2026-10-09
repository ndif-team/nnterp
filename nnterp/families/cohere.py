"""Cohere (``CohereForCausalLM``, Command-R).

``model.{embed_tokens, layers[i].{input_layernorm, self_attn, mlp}, norm}`` and
``lm_head``: Llama's names. The block is parallel: one LayerNorm (not RMS)
feeds both sublayers and the block returns ``x + attn + mlp``, so the
contributions are the two modules' outputs and the base holds. There is no
``post_attention_layernorm``. The attention runs the shared interface, after
an optional per-head ``q_norm`` / ``k_norm`` (``use_qk_norm``) and an
interleaved rotary.

The model multiplies the head's output by ``config.logit_scale`` to make the
logits, so the family defines ``project_on_vocab`` with that step, and the
logit lens on the last block equals ``logits``.

Aya Vision 32B (``aya_vision`` with a ``cohere`` text config) loads through this
family. It binds no tower keys, so that load has no ``vision`` or ``projector``.
"""

from typing import TYPE_CHECKING

import torch
from transformers.models.cohere.modeling_cohere import CohereAttention, CohereDecoderLayer, CohereMLP

from ..components import Attention, Layer, Mlp

if TYPE_CHECKING:
    from ..standardized import StandardizedTransformer

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """Cohere's parallel block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Cohere's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """Cohere's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {CohereDecoderLayer: Layer, CohereAttention: Attention, CohereMLP: Mlp}


def project_on_vocab(model: "StandardizedTransformer", hidden: torch.Tensor) -> torch.Tensor:
    """The logit lens as the model makes its logits: the final norm, ``lm_head``, then times ``logit_scale``."""
    return model.lm_head(model.norm(hidden)) * model.config.logit_scale
