"""Cohere (``CohereForCausalLM``, Command-R).

``model.{embed_tokens, layers[i].{input_layernorm, self_attn, mlp}, norm}`` and
``lm_head``: Llama's names. The block is parallel: one LayerNorm (not RMS)
feeds both sublayers and the block returns ``x + attn + mlp``, so the
contributions are the two modules' outputs and the base holds. There is no
``post_attention_layernorm``. The attention runs the shared interface, after
an optional per-head ``q_norm`` / ``k_norm`` (``use_qk_norm``) and an
interleaved rotary.

``CohereForCausalLM`` multiplies the head's output by ``config.logit_scale`` to
make the logits, so the family defines ``project_on_vocab`` with that step, and
the logit lens on the last block equals ``logits``. The Aya Vision and
Cohere2-Vision wrappers never scale, so the lens scales only on the causal-LM
classes.

Aya Vision 32B (``aya_vision`` with a ``cohere`` text config) loads through this
family with ``task="image-text-to-text"``: the text names bind on the wrapper's
``model.language_model.{embed_tokens, layers, norm}`` with ``lm_head`` at the root,
and the lens there is ``lm_head(norm(h))``, unscaled, as the wrapper's logits are.
The family binds no tower keys, so that load has no ``vision`` or ``projector``.
"""

from typing import TYPE_CHECKING

import torch
from transformers.models.cohere.modeling_cohere import CohereAttention, CohereDecoderLayer, CohereForCausalLM, CohereMLP
from transformers.models.cohere2.modeling_cohere2 import Cohere2ForCausalLM

from ..components import Attention, EProperty, Layer, Mlp, PARALLEL, Residual, unavailable

if TYPE_CHECKING:
    from ..standardized import StandardizedTransformer

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    # The same text model inside Aya Vision 32B's wrapper (AyaVisionForConditionalGeneration with a cohere text
    # config), loaded with task="image-text-to-text".
    "model.language_model.embed_tokens": "embed_tokens",
    "model.language_model.layers": "layers",
    "model.language_model.norm": "norm",
}


class Layer(Layer):
    """Cohere's parallel block; returns a bare tensor, so the base holds.

    A parallel block: there is no stream between the sublayers, and the MLP reads the same normed input
    as the attention.
    """

    layer_mid = unavailable(PARALLEL)

    @EProperty("input_layernorm.output", description="The normed stream entering the MLP: the one norm feeds both sublayers")
    def mlp_norm_output(self, value) -> Residual:
        return value



class Attention(Attention):
    """Cohere's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """Cohere's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {CohereDecoderLayer: Layer, CohereAttention: Attention, CohereMLP: Mlp}


def project_on_vocab(model: "StandardizedTransformer", hidden: torch.Tensor) -> torch.Tensor:
    """The logit lens as the model makes its logits: the final norm, ``lm_head``, then times ``logit_scale`` where the class applies it.

    ``CohereForCausalLM`` and ``Cohere2ForCausalLM`` multiply by the config's
    ``logit_scale``; ``AyaVisionForConditionalGeneration`` and
    ``Cohere2VisionForConditionalGeneration`` return the head's output as it is,
    though their ``text_config`` sets a scale, so neither does the lens on them.
    """
    logits = model.lm_head(model.norm(hidden))
    if isinstance(model._module, (CohereForCausalLM, Cohere2ForCausalLM)):
        return logits * model.config.logit_scale
    return logits
