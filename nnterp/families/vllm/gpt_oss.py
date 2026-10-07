"""GPT-OSS on vLLM (``vllm.model_executor.models.gpt_oss``).

vLLM's names are its own: ``model.embedding`` is ``embed_tokens`` and each
block's attention is ``attn`` (aliased to ``self_attn``), whose own ``attn``
child is vLLM's attention layer, as on every family. The block is fused but
takes the stream first: ``forward(hidden_states, positions, residual) ->
(hidden_states, residual)``. Its MLP is a mixture of experts whose output is
what the block adds; the routing is inside vLLM's fused MoE kernel.

The attention has a **sink** per head: a learned logit that joins the softmax
as one extra key column, inside vLLM's kernel, and is dropped afterwards. The
recomputed pattern does the same with the module's ``sinks``, so its rows sum
to less than one, as on transformers; ``attention_scores`` are the masked
scores before the sink joins them, also as on transformers.
"""

import torch
from vllm.model_executor.models.gpt_oss import MLPBlock, OAIAttention, TransformerBlock

from ...components import Pattern
from ...components.eproperty import DerivedEProperty
from ...components.vllm import Attention, FusedLayer, Mlp, on_decode_step

RENAME = {
    "model.embedding": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "attn": "self_attn",
}


def probabilities(self: "Attention") -> Pattern:
    """The pattern with the sink: the softmax over the scores plus the head's sink column, the sink column dropped."""
    scores = self.attention_scores
    sinks = self._module.sinks.to(torch.float32).reshape(1, -1, 1, 1).expand(*scores.shape[:-1], 1)
    combined = torch.cat([scores.float(), sinks], dim=-1)
    return combined.softmax(-1)[..., :-1].to(scores.dtype)


class Layer(FusedLayer):
    """The decoder block: ``forward(hidden_states, positions, residual) -> (hidden_states, residual)``."""

    HIDDEN = (0, "hidden_states")
    RESIDUAL = (2, "residual")


class Attention(Attention):
    """The attention, with a sink column in the softmax: the pattern's rows sum to less than one."""

    #: The pattern's rows sum to less than one: the sink takes the rest.
    SINK = True

    attention_probabilities = DerivedEProperty(
        probabilities,
        description="The attention pattern with the sink column dropped, [1, heads, query, key], recomputed; prefill only, read-only",
        unavailable=on_decode_step,
    )


class Mlp(Mlp):
    """The mixture of experts; its output is what the block adds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {TransformerBlock: Layer, OAIAttention: Attention, MLPBlock: Mlp}
