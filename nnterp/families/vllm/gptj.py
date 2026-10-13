"""GPT-J on vLLM (``vllm.model_executor.models.gpt_j``).

Its tree is transformers' (``transformer.{wte, h[i].{ln_1, attn, mlp},
ln_f}`` plus ``lm_head``). The block is parallel: it is called with the
positions and the residual stream, runs the attention and the MLP on the one
normed input, and returns the stream with both added. The head has a bias,
which `project_on_vocab` adds as the model does.
"""

from typing import TYPE_CHECKING

import torch

from vllm.model_executor.models.gpt_j import GPTJAttention, GPTJBlock, GPTJMLP

from ...components import PARALLEL, Residual, unavailable
from ...components.vllm import Attention, Flat, Layer, Mlp, project

if TYPE_CHECKING:
    from ...standardized_vllm import StandardizedVLLM

RENAME = {
    "transformer.wte": "embed_tokens",
    "transformer.h": "layers",
    "transformer.ln_f": "norm",
    "ln_1": "input_layernorm",
    "attn": "self_attn",
    # The sublayers' interiors: the output projection, the activation and the down projection under their standard names.
    "out_proj": "o_proj",
    "act": "act_fn",
    "fc_out": "down_proj",
}


class Layer(Layer):
    """The block: called with the positions and the residual stream, and returns the stream."""

    STREAM = 1

    #: A parallel block: no stream between the sublayers, and the one norm feeds both.
    layer_mid = unavailable(PARALLEL)

    @Flat("input_layernorm.output", description="The normed stream entering the MLP: the one norm feeds both sublayers, [1, tokens, hidden_size]")
    def mlp_norm_output(self, value) -> Residual:
        return value


class Attention(Attention):
    """The attention; its output is what the block adds, so the base holds."""


class Mlp(Mlp):
    """The MLP; its output is what the block adds, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {GPTJBlock: Layer, GPTJAttention: Attention, GPTJMLP: Mlp}


# -- the logit lens: the head has a bias ------------------------------------------------

def project_on_vocab(model: "StandardizedVLLM", hidden: torch.Tensor) -> torch.Tensor:
    """The final norm, then vLLM's logits processor over ``lm_head`` with its bias, as the model's own logits are."""
    return project(model, hidden, model.lm_head, model.lm_head.bias)


# -- sizes: what GPT-J's config calls them, as on transformers -----------------------------

from ..gptj import intermediate_size  # noqa: E402, F401
