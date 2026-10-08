"""GPT-BigCode on vLLM, through its transformers backend (``GPTBigCodeForCausalLM`` -> ``TransformersForCausalLM``).

vLLM has no GPT-BigCode of its own: it runs transformers' modules, so the
block is transformers' (see `nnterp.components.vllm.transformers_backend`),
with the stream ``[1, tokens, hidden]`` and the engine's attention layer
mounted as the attention's ``attn`` child. The tree is GPT-2's under
``model`` rather than ``transformer`` (the backend builds the bare
``GPTBigCodeModel``): ``model.{wte, wpe, h[i].{ln_1, attn, ln_2, mlp}, ln_f}``
and ``lm_head``. The block adds both contributions itself and returns the
stream. Under ``multi_query`` (every released checkpoint) the keys and
values are one head shared by every query head, as the engine's layer is
built. ``token_embeddings`` is ``wte``'s output, before the positions are added.
"""

from transformers.models.gpt_bigcode.modeling_gpt_bigcode import GPTBigCodeAttention, GPTBigCodeBlock, GPTBigCodeMLP

from ...components.vllm.transformers_backend import Attention, Layer, Mlp
from ..gpt_bigcode import intermediate_size, num_kv_heads  # noqa: F401  sizes the config spells its own way

RENAME = {
    "model.wte": "embed_tokens",
    "model.h": "layers",
    "model.ln_f": "norm",
    "ln_1": "input_layernorm",
    "attn": "self_attn",
    "ln_2": "post_attention_layernorm",
}


class Layer(Layer):
    """GPT-BigCode's block: called with the stream and returning it, so the base holds."""


class Attention(Attention):
    """GPT-BigCode's attention; its output is what the block adds, so the base holds."""


class Mlp(Mlp):
    """GPT-BigCode's MLP; its output is what the block adds, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {GPTBigCodeBlock: Layer, GPTBigCodeAttention: Attention, GPTBigCodeMLP: Mlp}
