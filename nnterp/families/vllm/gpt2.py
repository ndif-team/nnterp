"""GPT-2 on vLLM (``vllm.model_executor.models.gpt2``).

Its tree is transformers' (``transformer.{wte, wpe, h[i].{ln_1, attn, ln_2,
mlp}, ln_f}`` plus ``lm_head``) and its block is not fused: it is called with
the residual stream and returns it, adding both contributions itself.
"""

from typing import TYPE_CHECKING

from vllm.model_executor.models.gpt2 import GPT2Attention, GPT2Block, GPT2MLP

from ...components.vllm import Attention, Layer, Mlp

if TYPE_CHECKING:
    from ...standardized_vllm import StandardizedVLLM

RENAME = {
    "transformer.wte": "embed_tokens",
    "transformer.h": "layers",
    "transformer.ln_f": "norm",
    "ln_1": "input_layernorm",
    "attn": "self_attn",
    "ln_2": "post_attention_layernorm",
}


class Layer(Layer):
    """GPT-2's block: ``forward(hidden_states) -> hidden_states``, so the base holds."""


class Attention(Attention):
    """GPT-2's attention; its output is what the block adds, so the base holds."""


class Mlp(Mlp):
    """GPT-2's MLP; its output is what the block adds, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {GPT2Block: Layer, GPT2Attention: Attention, GPT2MLP: Mlp}


# -- sizes: what GPT-2's config calls them ------------------------------------------

def intermediate_size(model: "StandardizedVLLM") -> int:
    """The MLP width is ``n_inner``, ``None`` meaning four times the hidden size."""
    return model.config.n_inner or 4 * model.hidden_size
