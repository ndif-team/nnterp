"""Llama 4, text (``Llama4ForCausalLM``) on vLLM (``vllm.model_executor.models.llama4``).

Llama's names and Llama's fused block: it takes and returns the stream as
``(hidden_states, residual)``. The feed-forward is ``feed_forward``, aliased to
``mlp`` as on transformers: a dense MLP on some blocks, on the others
(every ``interleave_moe_layer_step``-th) a mixture of experts whose output,
the shared expert's plus the routed experts', is what the block adds. The
routing is inside vLLM's fused MoE kernel and has no values here. The shared
expert (``shared_expert``, aliased to ``shared_experts``) is the dense MLP's
class, so it is an `Mlp` too; its ``mlp_output`` is unavailable.

The attention is iRoPE's, as on transformers: a NoPE block
(``no_rope_layers[i] == 0``) has no rotary embedding and no query and key
norm and attends over the whole prefix; the others rotate, then L2-norm their
queries and keys, then attend within chunks of ``attention_chunk_size``
tokens (vLLM's ``ChunkedLocalAttention``), so the recomputed scores mask the
keys of other chunks too. The queries and keys served are the ones the
attention layer receives, after all of that, with one change: vLLM permutes
each head's query and key channels at load (``q_proj`` and ``k_proj`` rows,
so that its rotary embedding can rotate halves where transformers rotates
adjacent pairs), and the values put them back in transformers' order, and an
edit back in vLLM's.

One difference from transformers is in vLLM's model, not here: vLLM applies
the NoPE blocks' attention temperature tuning only when the generation
config says ``attn_temperature_tuning`` or ``max_model_len`` exceeds 32K,
whatever the model config says (pass
``override_generation_config={"attn_temperature_tuning": True}`` to match
transformers). The tuning leaves the queries of the first 8191 positions
unscaled either way.
"""

import torch

from vllm.model_executor.models.llama import LlamaMLP
from vllm.model_executor.models.llama4 import Llama4Attention, Llama4DecoderLayer, Llama4MoE

from ...components import Keys, Queries, Residual
from ...components.eproperty import DerivedEProperty
from ...components.vllm import Attention, Flat, FusedLayer, Mlp, on_decode_step
from ...components.vllm.attention import scores
from ..llama4_text import intermediate_size  # noqa: F401  the dense MLP's width, as on transformers

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
    "feed_forward": "mlp",
    "shared_expert": "shared_experts",
}


def _not_a_block_feed_forward(envoy) -> str | None:
    if envoy.path.rsplit(".", 1)[-1] == "feed_forward":
        return None
    return "on vLLM this is the shared expert inside the mixture of experts; what the block adds is the mixture's output, at layers[i].mlp"


def chunked_scores(self: "Attention") -> torch.Tensor:
    """The base's recomputed scores, with the keys of another chunk at ``-inf`` on a chunked block."""
    values = scores(self)
    chunk = getattr(self.attn._module, "attention_chunk_size", None)
    if chunk:
        position = torch.arange(values.shape[-1], device=values.device) // chunk
        values = values.masked_fill(position[:, None] != position[None, :], float("-inf"))
    return values


def pairs_first(value: torch.Tensor) -> torch.Tensor:
    """A head's channels from vLLM's load-time order (the even channels, then the odd ones) back to transformers' order."""
    return value.unflatten(-1, (2, -1)).transpose(-1, -2).flatten(-2)


def halves_first(value: torch.Tensor) -> torch.Tensor:
    """`pairs_first`'s inverse: transformers' order of a head's channels to vLLM's."""
    return value.unflatten(-1, (-1, 2)).transpose(-1, -2).flatten(-2)


class Interleaved(Flat):
    """A `Flat` over queries or keys whose channels vLLM permuted at load: served in transformers' order, handed back in vLLM's."""

    def _rows(self, obj, value: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        return super()._rows(obj, halves_first(value), rows)


class Layer(FusedLayer):
    """The decoder block: ``forward(positions, hidden_states, residual) -> (hidden_states, residual)``."""


class Attention(Attention):
    """The attention; its output is what the block adds. On a chunked block a query sees only its own chunk's keys."""

    attention_scores = DerivedEProperty(
        chunked_scores,
        description="The scores entering the softmax, [1, heads, query, key], recomputed from the queries and keys; prefill only, read-only",
        unavailable=on_decode_step,
    )

    @Interleaved("attn.inputs", select=0, heads="first", description="The queries entering vLLM's attention layer, [1, heads, tokens, head_dim], in transformers' channel order")
    def attention_queries(self, value: torch.Tensor) -> Queries:
        """The queries the attention layer receives, each head's channels put back in transformers' order."""
        return pairs_first(value)

    @Interleaved("attn.inputs", select=1, heads="first", description="The keys entering vLLM's attention layer, [1, kv_heads, tokens, head_dim], in transformers' channel order")
    def attention_keys(self, value: torch.Tensor) -> Keys:
        """This step's keys as the attention layer receives them, each head's channels put back in transformers' order."""
        return pairs_first(value)


class Mlp(Mlp):
    """The dense MLP or the mixture of experts; its output is what the block adds, but on the shared expert."""

    @Flat("output", description="What the MLP adds to the residual stream, [1, tokens, hidden]", unavailable=_not_a_block_feed_forward)
    def mlp_output(self, value) -> Residual:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Llama4DecoderLayer: Layer, Llama4Attention: Attention, LlamaMLP: Mlp, Llama4MoE: Mlp}
