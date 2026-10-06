"""GPT-BigCode (``GPTBigCodeForCausalLM``): StarCoder, SantaCoder.

GPT-2's tree and GPT-2's block: ``transformer.{wte, wpe, h[i].{ln_1, attn, ln_2,
mlp}, ln_f}`` and ``lm_head``, the residual added in the block, which returns a bare
tensor. The attention projects queries, keys and values with one fused ``c_attn``
and runs the shared eager forward. Under ``multi_query`` (every released checkpoint)
the keys and values are one head shared by every query head, so ``num_kv_heads`` is
1. Queries, keys and values are split views of the one ``c_attn`` tensor. The MLP is
``n_inner`` wide, ``4 * hidden_size`` when that is ``None``.
"""

from typing import TYPE_CHECKING

from transformers.models.gpt_bigcode.modeling_gpt_bigcode import GPTBigCodeAttention, GPTBigCodeBlock, GPTBigCodeMLP

from ..components import Attention, Layer, Mlp

if TYPE_CHECKING:
    from ..standardized import StandardizedTransformer

RENAME = {
    "transformer.wte": "embed_tokens",
    "transformer.h": "layers",
    "transformer.ln_f": "norm",
    "ln_1": "input_layernorm",
    "attn": "self_attn",
    "ln_2": "post_attention_layernorm",
}


class Layer(Layer):
    """GPT-BigCode's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """GPT-BigCode's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """GPT-BigCode's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {GPTBigCodeBlock: Layer, GPTBigCodeAttention: Attention, GPTBigCodeMLP: Mlp}


# -- sizes -------------------------------------------------------------------------

def num_kv_heads(model: "StandardizedTransformer") -> int:
    """One key/value head under ``multi_query``; else every head."""
    return 1 if model.config.multi_query else model.num_heads


def intermediate_size(model: "StandardizedTransformer") -> int:
    """The MLP width is ``n_inner``, ``4 * hidden_size`` when ``None``."""
    return model.config.n_inner or 4 * model.hidden_size
