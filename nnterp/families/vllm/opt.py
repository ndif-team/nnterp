"""OPT on vLLM (``vllm.model_executor.models.opt``).

Its tree is transformers' (``model.decoder.{embed_tokens, embed_positions,
layers[i].{self_attn_layer_norm, self_attn, final_layer_norm, fc1, fc2},
final_layer_norm}``). The block is plain: called with the residual stream
alone, it returns the stream. As on transformers there is no MLP module:
``fc1`` and ``fc2`` sit on the block, so ``mlp`` and ``mlp_output`` do not
exist here either, and what the feed-forward path adds is ``fc2``'s output
(vLLM's linear layers return ``(output, bias)``, the bias already added: read
``layers[i].fc2.output[0]``).

transformers scales the queries by ``head_dim**-0.5`` at ``q_proj``'s output;
vLLM leaves them unscaled and its attention layer applies the scale. So
``attention_queries`` is served times that scale, the pre-scaled queries
transformers serves, as a computed copy divided back on a write, and the
recomputed scores are ``queries @ keys.T`` with no further scale.

When the checkpoint ties its embeddings (OPT-125m and most of the family),
vLLM's ``lm_head`` *is* ``embed_tokens``: the same module under both names.
On a checkpoint with ``do_layer_norm_before`` false (OPT-350m) the block
norms after each add, and the stream is not ``layer_input`` plus the two
contributions.
"""

import torch

from vllm.model_executor.models.opt import OPTAttention, OPTDecoderLayer

from ...components import Pattern, Queries
from ...components.eproperty import DerivedEProperty
from ...components.vllm import Attention, Flat, Layer, Mlp, on_decode_step
from ..opt import intermediate_size  # noqa: F401  the width of fc1/fc2 is ffn_dim, as on transformers

RENAME = {
    "model.decoder.embed_tokens": "embed_tokens",
    "model.decoder.layers": "layers",
    "model.decoder.final_layer_norm": "norm",
    "self_attn_layer_norm": "input_layernorm",
}


def query_scale(envoy) -> float:
    """``head_dim**-0.5``, which transformers applies to the queries and vLLM's attention layer to the scores."""
    return envoy._module.scaling


def scores(self: "Attention") -> Pattern:
    """The causal scores, ``[1, heads, query, key]``: the pre-scaled queries times the keys, later keys at ``-inf``."""
    values = self.attention_queries @ self.attention_keys.transpose(-1, -2)
    position = torch.arange(values.shape[-1], device=values.device)
    return values.masked_fill(position[None, :] > position[:, None], float("-inf"))


class Layer(Layer):
    """OPT's block: ``forward(hidden_states) -> hidden_states``, so the base holds."""


class Attention(Attention):
    """OPT's attention: its output is what the block adds; its queries are served pre-scaled, as on transformers."""

    attention_scores = DerivedEProperty(
        scores,
        description="The scores entering the softmax, [1, heads, query, key], recomputed from the pre-scaled queries and the keys; prefill only, read-only",
        unavailable=on_decode_step,
    )

    @Flat(
        "attn.inputs", select=0, heads="first", factor=query_scale,
        description="The queries entering vLLM's attention layer times head_dim**-0.5, as transformers scales them, [1, heads, tokens, head_dim]",
    )
    def attention_queries(self, value: torch.Tensor) -> Queries:
        return value


class Mlp(Mlp):
    """OPT has no MLP module; this class exists so the family declares one, never instantiated."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``. No MLP module to key.
ENVOYS = {OPTDecoderLayer: Layer, OPTAttention: Attention}
