"""Gemma 2 on vLLM (``vllm.model_executor.models.gemma2``).

Llama's names and a fused block, with Gemma-2's sandwich norms: what the
block adds is each sublayer's *post* norm's output, not the module's, as on
transformers, so the contributions point at the sibling norms.

Two things are not transformers': there is no ``lm_head`` module (the logits
processor unembeds with ``embed_tokens``' weight, which the checkpoint ties),
so `project_on_vocab` is defined here; and the ``sqrt(hidden_size)`` scaling
of the embeddings is applied by the model after ``embed_tokens``, not inside
it, so ``token_embeddings`` is the unscaled lookup and
``layers[0].layer_input`` the scaled stream. The final logit softcapping is
applied by vLLM's logits processor, which both `logits` and
`project_on_vocab` go through.
"""

from typing import TYPE_CHECKING

import torch

from vllm.model_executor.models.gemma2 import Gemma2Attention, Gemma2DecoderLayer, Gemma2MLP

from ...components import Residual
from ...components.vllm import Attention, Flat, FusedLayer, Mlp, project

if TYPE_CHECKING:
    from ...standardized_vllm import StandardizedVLLM

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(FusedLayer):
    """Gemma-2's decoder block: ``forward(positions, hidden_states, residual) -> (hidden_states, residual)``."""


class Attention(Attention):
    """Gemma-2's attention: what reaches the residual stream is the post-attention norm's output."""

    @Flat(
        "../post_attention_layernorm.output",
        description="What the attention adds to the residual stream: the post-attention norm's output, [1, tokens, hidden]",
    )
    def attention_output(self, value) -> Residual:
        return value


class Mlp(Mlp):
    """Gemma-2's MLP: what reaches the residual stream is the post-feedforward norm's output."""

    @Flat(
        "../post_feedforward_layernorm.output",
        description="What the MLP adds to the residual stream: the post-feedforward norm's output, [1, tokens, hidden]",
    )
    def mlp_output(self, value) -> Residual:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Gemma2DecoderLayer: Layer, Gemma2Attention: Attention, Gemma2MLP: Mlp}


# -- the logit lens: no lm_head module ------------------------------------------------

def project_on_vocab(model: "StandardizedVLLM", hidden: torch.Tensor) -> torch.Tensor:
    """The final norm, then vLLM's logits processor over ``embed_tokens``, which is the unembedding here."""
    return project(model, hidden, model.embed_tokens)
