"""Cohere Command R on vLLM (``vllm.model_executor.models.commandr``).

vLLM runs Command R (``cohere``) and Command R7B / Aya (``cohere2``) through
one implementation. The names are Llama's. The block is parallel and is not
fused, whatever its signature says: it is called ``forward(positions,
hidden_states, residual)``, ignores the ``residual`` it is handed, runs the
attention and the MLP on the one normed input, and returns ``(hidden_states,
residual)`` with the *whole* stream first. There is no ``lm_head`` module
(the logits processor unembeds with ``embed_tokens``' weight, which the
checkpoint ties, and applies the model's ``logit_scale``), so
`project_on_vocab` is defined here.
"""

from typing import TYPE_CHECKING

import torch
from vllm.model_executor.models.commandr import CohereAttention, CohereDecoderLayer, CohereMLP

from ...components import PARALLEL, Residual, unavailable
from ...components.vllm import Attention, Flat, Layer, Mlp, project

if TYPE_CHECKING:
    from ...standardized_vllm import StandardizedVLLM

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """The block: called with the positions and the stream, parallel, returning ``(stream, residual)``."""

    STREAM = 1
    returns_tuple = True

    #: A parallel block: no stream between the sublayers, and the one norm feeds both (called with the residual, it returns the normed half first).
    layer_mid = unavailable(PARALLEL)

    @Flat("input_layernorm.output", description="The normed stream entering the MLP: the one norm feeds both sublayers, [1, tokens, hidden_size]")
    def mlp_norm_output(self, value) -> Residual:
        return value


class Attention(Attention):
    """The attention; its output is what the block adds, so the base holds."""


class Mlp(Mlp):
    """The MLP; its output is what the block adds, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {CohereDecoderLayer: Layer, CohereAttention: Attention, CohereMLP: Mlp}


# -- the logit lens: no lm_head module ------------------------------------------------

def project_on_vocab(model: "StandardizedVLLM", hidden: torch.Tensor) -> torch.Tensor:
    """The final norm, then vLLM's logits processor over ``embed_tokens``, which is the unembedding here."""
    return project(model, hidden, model.embed_tokens)
