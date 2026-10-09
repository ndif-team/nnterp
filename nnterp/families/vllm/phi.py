"""Phi-1 and Phi-2 on vLLM (``vllm.model_executor.models.phi``).

Its tree is transformers' (``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, mlp}, final_layernorm}`` plus ``lm_head``). The block is parallel:
it is called with the positions and the residual stream, runs the attention
and the MLP on the one normed input, and returns the stream with both added.
The head has a bias, which `project_on_vocab` adds as the model does.
"""

from typing import TYPE_CHECKING

import torch

from vllm.model_executor.models.phi import PhiAttention, PhiLayer, PhiMLP

from ...components.vllm import Attention, Layer, Mlp, project

if TYPE_CHECKING:
    from ...standardized_vllm import StandardizedVLLM

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.final_layernorm": "norm",
}


class Layer(Layer):
    """The block: called with the positions and the residual stream, and returns the stream."""

    STREAM = 1


class Attention(Attention):
    """The attention; its output is what the block adds, so the base holds."""


class Mlp(Mlp):
    """The MLP; its output is what the block adds, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {PhiLayer: Layer, PhiAttention: Attention, PhiMLP: Mlp}


# -- the logit lens: the head has a bias ------------------------------------------------

def project_on_vocab(model: "StandardizedVLLM", hidden: torch.Tensor) -> torch.Tensor:
    """The final norm, then vLLM's logits processor over ``lm_head`` with its bias, as the model's own logits are."""
    return project(model, hidden, model.lm_head, model.lm_head.bias)
