"""DBRX on vLLM (``vllm.model_executor.models.dbrx``).

Transformers' tree: ``transformer.{wte, blocks[i].{norm_attn_norm.{norm_1,
attn, norm_2}, ffn}, norm_f}`` and ``lm_head``, aliased as on transformers.
The block is plain: it is called with the positions and the residual stream
and returns the stream. Its ``norm_attn_norm`` norms, runs the attention,
adds the residual and norms again, returning ``(normed, stream)``; the block
adds the FFN's output to that stream. So the attention module's output is its
contribution, and so is the FFN's, a mixture of experts whose routing is
inside vLLM's fused MoE kernel and has no values here.
"""

from vllm.model_executor.models.dbrx import DbrxAttention, DbrxBlock, DbrxMoE

from ...components.vllm import Attention, Layer, Mlp
from ..dbrx import intermediate_size  # noqa: F401  the experts' width, as on transformers

RENAME = {
    "transformer.wte": "embed_tokens",
    "transformer.blocks": "layers",
    "transformer.norm_f": "norm",
    "norm_attn_norm.attn": "self_attn",
    "norm_attn_norm.norm_1": "input_layernorm",
    "norm_attn_norm.norm_2": "post_attention_layernorm",
    "ffn": "mlp",
}


class Layer(Layer):
    """The block: called with the positions and the residual stream, and returns the stream."""

    STREAM = 1


class Attention(Attention):
    """The attention; its output is what ``norm_attn_norm`` adds, so the base holds."""


class Mlp(Mlp):
    """The mixture of experts; its output is what the block adds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {DbrxBlock: Layer, DbrxAttention: Attention, DbrxMoE: Mlp}
