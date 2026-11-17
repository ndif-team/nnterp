"""Phi-1 and Phi-2 on vLLM, through its transformers backend (``PhiForCausalLM`` -> ``TransformersForCausalLM``).

vLLM runs transformers' modules (see `nnterp.components.vllm.transformers_backend`):
``model.{embed_tokens, layers[i].{input_layernorm, self_attn, mlp},
final_layernorm}`` plus ``lm_head``. The block is parallel: it runs the
attention and the MLP on the one normed input and returns the stream with
both added. The head has a bias, which `project_on_vocab` adds as the model does.
"""

from typing import TYPE_CHECKING

import torch
from transformers.models.phi.modeling_phi import PhiAttention, PhiDecoderLayer, PhiMLP

from ...components import PARALLEL, Residual, unavailable
from ...components.vllm import Flat, project
from ...components.vllm.transformers_backend import Attention, Layer, Mlp

if TYPE_CHECKING:
    from ...standardized_vllm import StandardizedVLLM

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.final_layernorm": "norm",
    # The sublayers' interiors: the output projection, the activation and the down projection under their standard names.
    "dense": "o_proj",
    "activation_fn": "act_fn",
    "fc2": "down_proj",
}


class Layer(Layer):
    """Phi's block: called with the stream and returning it, parallel."""

    #: A parallel block: no stream between the sublayers, and the one norm feeds both.
    layer_mid = unavailable(PARALLEL)

    @Flat("input_layernorm.output", batch=True, description="The normed stream entering the MLP: the one norm feeds both sublayers, [1, tokens, hidden_size]")
    def mlp_norm_output(self, value) -> Residual:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {PhiDecoderLayer: Layer, PhiAttention: Attention, PhiMLP: Mlp}


# -- the logit lens: the head has a bias ------------------------------------------------

def project_on_vocab(model: "StandardizedVLLM", hidden: torch.Tensor) -> torch.Tensor:
    """The final norm, then vLLM's logits processor over ``lm_head`` with its bias, as the model's own logits are."""
    return project(model, hidden, model.lm_head, model.lm_head.bias)
