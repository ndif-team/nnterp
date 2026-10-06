"""DeepSeek-V3.2 (``DeepseekV32ForCausalLM``).

Llama's tree and Llama's block: ``model.{embed_tokens, layers[i].{input_layernorm,
self_attn, post_attention_layernorm, mlp}, norm}`` and ``lm_head``, the residual
added in the block, a bare tensor returned. Multi-head latent attention as in V3, with
DeepSeek Sparse Attention on top: each attention module carries an ``indexer`` (a
lightning indexer with its own small projections) that scores every key for every
query and keeps the ``index_topk`` best. Under eager the selection is folded into the
attention mask (every unselected key set to the dtype's minimum) before the shared
``attention_interface`` call, so the interior is the base's, read on the interface:

- ``attention_scores`` are the scaled scores with that sparse mask added, so an
  unselected key reads as the dtype's minimum, like a future one.
- ``attention_probabilities`` is the dense ``[batch, heads, query, key]`` pattern of
  the sparse attention: exactly zero outside each query's selected keys, rows summing
  to one over the selection. A written pattern is used as written, so a write can put
  weight on keys the indexer did not select.

While a prompt is no longer than ``index_topk`` (2048 on the released checkpoints) the
indexer selects every causal key and the pattern is the dense causal one. The indexer's
selection is ``self_attn.indexer.output``, the ``[batch, query, index_topk]`` int32 key
indices; it has no standard name. The first ``first_k_dense_replace`` blocks (per
``mlp_layer_types``) have a dense MLP, the rest a mixture of experts. The sizes are
DeepSeek-V2's.
"""

from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32Attention, DeepseekV32DecoderLayer, DeepseekV32MLP, DeepseekV32MoE

from ..components import Attention, Layer, Mlp, Moe
from .deepseek_v2 import head_dim, qk_head_dim  # noqa: F401  the same latent attention: the same sizes

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "gate": "router",
}


class Layer(Layer):
    """DeepSeek-V3.2's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """DeepSeek-V3.2's sparse latent attention; the indexer's selection is folded into the mask before the shared eager forward, so the base holds."""


class Mlp(Mlp):
    """DeepSeek's dense MLP or mixture of experts; both return the hidden states, and the residual is added in the block."""


class Moe(Moe, Mlp):
    """DeepSeek-V3.2's mixture of experts: DeepSeek-V3's."""

    SCORING = "sigmoid"


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {DeepseekV32DecoderLayer: Layer, DeepseekV32Attention: Attention, DeepseekV32MLP: Mlp, DeepseekV32MoE: Moe}
