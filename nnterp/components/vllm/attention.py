"""`Attention`: a vLLM attention module, its contribution, what crosses the engine's attention layer, and the pattern."""

from __future__ import annotations

import torch
from nnsight.intervention.envoy import Envoy
from nnsight.intervention.interleaver import Mediator

from .. import attention as standard
from ..attention import HeadOutputs, Keys, Pattern, Queries, Values
from ..eproperty import DerivedEProperty
from ..layer import Residual
from .flat import Flat

#: Why the scores and the pattern are not there on a decode step: they are recomputed from the step's own rows.
DECODE_STEP = (
    "recomputed from this step's queries and keys, and on a decode step the earlier tokens' keys are in vLLM's "
    "cache, which only its kernel reads; read it on the prefill"
)


def on_decode_step(envoy: Envoy) -> str | None:
    """`DECODE_STEP` when a read made now would be served a decode step of the request, else ``None``.

    A request's first visit to its attention layer is the prefill, every
    prompt token at once, and each later one a decode step of one token. The
    worker knows which visit a read would ask for before anything is read,
    so the value refuses there, by name. Outside a trace there is no step to
    ask about and the answer is the prefill's: `support` lists it available.
    """
    try:
        mediator = Mediator.current(envoy.path)
    except ValueError:
        return None
    return DECODE_STEP if mediator.wanted(f"{envoy.attn.path}.input") else None


def scores(self: "Attention") -> Pattern:
    """The scaled, masked scores the kernel's softmax runs on, ``[1, heads, query, key]``, recomputed.

    ``queries @ keys.T * scale``, softcapped where the layer softcaps, plus
    the ALiBi bias where the layer has slopes, with the keys a query may not
    see (later tokens, and tokens past the layer's sliding window) at
    ``-inf``. The scale, the softcap, the slopes and the window are the ones
    vLLM's attention layer was built with. The ALiBi bias is each head's
    slope times how far behind the query the key is; transformers' families
    write the same bias from another origin, which moves a query's scores by
    one constant and leaves the pattern alone.
    """
    queries, keys = self.attention_queries, self.attention_keys
    keys = keys.repeat_interleave(queries.shape[1] // keys.shape[1], dim=1)  # grouped-query: one key head per group
    layer = self.attn._module
    values = queries @ keys.transpose(-1, -2) * layer.impl.scale
    if layer.impl.logits_soft_cap:
        values = layer.impl.logits_soft_cap * torch.tanh(values / layer.impl.logits_soft_cap)
    position = torch.arange(values.shape[-1], device=values.device)
    behind = position[:, None] - position[None, :]  # how far behind the query each key is
    slopes = getattr(layer.impl, "alibi_slopes", None)
    if slopes is not None:
        values = values - slopes.to(values)[:, None, None] * behind
    hidden = behind < 0
    if layer.sliding_window is not None:
        hidden = hidden | (behind >= layer.sliding_window)
    return values.masked_fill(hidden, float("-inf"))


def probabilities(self: "Attention") -> Pattern:
    """The attention pattern, ``[1, heads, query, key]``: the softmax of `attention_scores`, in float32 then the model's dtype."""
    values = self.attention_scores
    return values.float().softmax(-1).to(values.dtype)


class Attention(standard.Attention):
    """A vLLM attention module: its contribution, what goes into and comes out of the engine's attention layer, and the pattern.

    The module projects (and, on a rotary family, rotates) its queries, keys
    and values and calls vLLM's attention layer on them, ``self.attn(q, k,
    v)``, which returns the per-head outputs the output projection then
    mixes. Those four are the ``attn`` child's inputs and output, each
    ``[tokens, heads * head_dim]``, served with the heads split out by the
    module's own ``head_dim``. A family whose attention module names that
    child, or its head width, another way points the values at it.

    They are this step's rows: on a decode step the keys and values are the
    new token's alone, and the earlier ones are in the engine's cache, which
    the kernel reads for itself.

    The scores and the pattern are computed inside the kernel, which serves
    nothing between its inputs and its output, so here they are *recomputed*
    from the queries and keys, with the layer's own scale, softcap and
    window. That makes them read-only (the kernel never takes a pattern: edit
    the queries or keys to change what a head attends to, or its head outputs
    to change what it wrote) and there on the prefill only, where a step
    holds every key; a decode step raises `Unavailable` (`on_decode_step`).
    """

    attention_scores = DerivedEProperty(
        scores,
        description="The scores entering the softmax, [1, heads, query, key], recomputed from the queries and keys; prefill only, read-only",
        unavailable=on_decode_step,
    )
    attention_probabilities = DerivedEProperty(
        probabilities,
        description="The attention pattern, [1, heads, query, key], recomputed from the queries and keys; prefill only, read-only",
        unavailable=on_decode_step,
    )

    @Flat("attn.inputs", select=0, heads="first", description="The queries entering vLLM's attention layer, [1, heads, tokens, head_dim]")
    def attention_queries(self, value: torch.Tensor) -> Queries:
        """The queries the attention layer receives, after the projection, any query norm and the rotary embedding."""
        return value

    @Flat("attn.inputs", select=1, heads="first", description="The keys entering vLLM's attention layer, [1, kv_heads, tokens, head_dim]")
    def attention_keys(self, value: torch.Tensor) -> Keys:
        """This step's keys as the attention layer receives them; ``num_kv_heads`` wide under grouped-query attention."""
        return value

    @Flat("attn.inputs", select=2, heads="first", description="The values entering vLLM's attention layer, [1, kv_heads, tokens, head_dim]")
    def attention_values(self, value: torch.Tensor) -> Values:
        """This step's values as the attention layer receives them; ``num_kv_heads`` wide under grouped-query attention."""
        return value

    @Flat("attn.output", heads="last", description="The per-head outputs before the output projection, [1, tokens, heads, head_dim]")
    def attention_head_outputs(self, value: torch.Tensor) -> HeadOutputs:
        """Each head's output as the attention layer returns it, before the output projection mixes them."""
        return value

    @Flat("output", description="What the attention adds to the residual stream, [1, tokens, hidden]")
    def attention_output(self, value: torch.Tensor) -> Residual:
        return value
