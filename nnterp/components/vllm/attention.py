"""`Attention`: a vLLM attention module, its contribution, what crosses the engine's attention layer, and the pattern."""

from __future__ import annotations

from functools import partial
from typing import Any

import torch
from nnsight.intervention.envoy import Envoy
from nnsight.intervention.interleaver import Mediator

from .. import attention as standard
from ..attention import HeadOutputs, Keys, Pattern, Queries, Values, rotary_reason
from ..eproperty import DerivedEProperty, Unavailable, unavailable
from ..layer import Residual
from ..standard import absent
from .flat import Flat, on_transformers_backend

#: The call every vLLM attention module makes to rotate its queries and keys, before its attention layer:
#: ``q, k = self.rotary_emb(positions, q, k)``. Read as an operation of the module's own forward, because vLLM
#: builds one rotary embedding per configuration and every block that rotates the same way holds that one module,
#: whose own inputs cannot tell the blocks apart.
ROTARY = "self_rotary_emb_0"


def rotary_head_size(envoy: Envoy) -> int:
    """The width of a head as the block's rotary embedding views its rows: the module's ``head_size``."""
    return envoy._module.rotary_emb.head_size

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
    """The scaled, masked scores the kernel's softmax runs on, ``[1, num_heads, query, key]``, recomputed.

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
    """The attention pattern, ``[1, num_heads, query, key]``: the softmax of `attention_scores`, in float32 then the model's dtype."""
    values = self.attention_scores
    return values.float().softmax(-1).to(values.dtype)


class Attention(standard.Attention):
    """A vLLM attention module: its contribution, what goes into and comes out of the engine's attention layer, and the pattern.

    The module projects (and, on a rotary family, rotates) its queries, keys
    and values and calls vLLM's attention layer on them, ``self.attn(q, k,
    v)``, which returns the per-head outputs the output projection then
    mixes. Those four are the ``attn`` child's inputs and output, each
    ``[tokens, num_heads * head_dim]``, served with the heads split out by the
    module's own ``head_dim``. A family whose attention module names that
    child, or its head width, another way points the values at it.

    They are this step's rows: on a decode step the keys and values are the
    new token's alone, and the earlier ones are in the engine's cache, which
    the kernel reads for itself.

    The queries and keys before the rotary embedding are the arguments of
    the module's rotary call (`ROTARY`), after any query and key norm, in
    the same layouts, and `rotate` calls that rotary embedding on any tensor
    laid out like them. vLLM's rotary kernel rotates its arguments in place,
    and they are split views of the fused ``qkv`` projection, so these are
    private copies like every vLLM value, taken before the kernel runs.
    `no_rotary` is the one place that decides a block has no rotary.

    The scores and the pattern are computed inside the kernel, which serves
    nothing between its inputs and its output, so here they are *recomputed*
    from the queries and keys, with the layer's own scale, softcap and
    window. That makes them read-only (the kernel never takes a pattern: edit
    the queries or keys to change what a head attends to, or its head outputs
    to change what it wrote) and there on the prefill only, where a step
    holds every key; a decode step raises `Unavailable` (`on_decode_step`).
    """

    interface = unavailable(
        "vLLM's attention layer is a kernel that writes this step's keys and values into the engine's cache as it "
        "runs, so it cannot be called again on other tensors"
    )

    attention_scores = DerivedEProperty(
        scores,
        description="The scores entering the softmax, [1, num_heads, query, key], recomputed from the queries and keys; prefill only, read-only",
        unavailable=on_decode_step,
    )
    attention_probabilities = DerivedEProperty(
        probabilities,
        description="The attention pattern, [1, num_heads, query, key], recomputed from the queries and keys; prefill only, read-only",
        unavailable=on_decode_step,
    )

    @Flat(
        f"source.{ROTARY}.inputs", select=1, heads="first", head_dim=rotary_head_size, unavailable=rotary_reason,
        description="The queries entering the rotary embedding, [1, num_heads, tokens, head_dim]",
    )
    def attention_queries_prerope(self, value: torch.Tensor) -> Queries:
        """The queries the rotary embedding receives: after the projection and any query norm, before the rotation.

        `rotate` at the positions they run at gives `attention_queries`
        where the block does nothing more between the two. Assign or edit in
        place; the block rotates what is written for the position it is
        written at, so a query moved from another position or prompt is one
        assignment.
        """
        return value

    @Flat(
        f"source.{ROTARY}.inputs", select=2, heads="first", head_dim=rotary_head_size, unavailable=rotary_reason,
        description="The keys entering the rotary embedding, [1, num_kv_heads, tokens, head_dim]",
    )
    def attention_keys_prerope(self, value: torch.Tensor) -> Keys:
        """This step's keys as the rotary embedding receives them; ``num_kv_heads`` wide under grouped-query attention."""
        return value

    @Flat("attn.inputs", select=0, heads="first", description="The queries entering vLLM's attention layer, [1, num_heads, tokens, head_dim]")
    def attention_queries(self, value: torch.Tensor) -> Queries:
        """The queries the attention layer receives, after the projection, any query norm and the rotary embedding."""
        return value

    @Flat("attn.inputs", select=1, heads="first", description="The keys entering vLLM's attention layer, [1, num_kv_heads, tokens, head_dim]")
    def attention_keys(self, value: torch.Tensor) -> Keys:
        """This step's keys as the attention layer receives them; ``num_kv_heads`` wide under grouped-query attention."""
        return value

    @Flat("attn.inputs", select=2, heads="first", description="The values entering vLLM's attention layer, [1, num_kv_heads, tokens, head_dim]")
    def attention_values(self, value: torch.Tensor) -> Values:
        """This step's values as the attention layer receives them; ``num_kv_heads`` wide under grouped-query attention."""
        return value

    @Flat("attn.output", heads="last", description="The per-head outputs before the output projection, [1, tokens, num_heads, head_dim]")
    def attention_head_outputs(self, value: torch.Tensor) -> HeadOutputs:
        """Each head's output as the attention layer returns it, before the output projection mixes them."""
        return value

    @Flat(
        "o_proj.input", batch=on_transformers_backend, heads="last", unavailable=partial(absent, name="o_proj", where="attention"),
        description="What the output projection receives, [1, tokens, num_heads, head_dim]",
    )
    def attention_premix(self, value: torch.Tensor) -> HeadOutputs:
        """The output projection's input split by head, after any gate the attention applies; the head outputs where there is none."""
        return value

    @Flat("output", description="What the attention adds to the residual stream, [1, tokens, hidden_size]")
    def attention_output(self, value: torch.Tensor) -> Residual:
        return value

    # -- the rotary embedding -------------------------------------------------------

    def rotate(self, x: torch.Tensor, positions: Any) -> torch.Tensor:
        """``x`` rotated by this block's rotary embedding at absolute ``positions``, as the block rotates its queries and keys.

        Args:
            x: ``[..., tokens, head_dim]``: queries or keys the way
                `attention_queries_prerope` / `attention_keys_prerope` serve them.
            positions: Each row's absolute position, ``[tokens]`` (a tensor, a
                list or a ``range``). On an M-RoPE rotary (the Qwen-VL text
                models) these are text positions, the same on its three rows;
                a ``[3, tokens]`` tensor gives each row its own.

        The block's ``rotary_emb``, vLLM's own module for the family (its rope
        scaling, partial and interleaved rotations included), is called on a
        private copy of ``x``, since its kernel rotates in place, so
        ``rotate(attention_queries_prerope, positions)`` is the block's
        `attention_queries` bit for bit where the block does nothing more
        between its rotary and its attention. It runs in the engine's dtype
        and on its device and returns ``x``'s. The module is the engine's, so
        it is called inside a trace only, like `project_on_vocab`.

        Raises:
            Unavailable: the block applies no rotary embedding (`no_rotary`).
            RuntimeError: outside a trace.
        """
        reason = self.no_rotary()
        if reason:
            raise Unavailable(f"{self.path}.rotate is not available: {reason}")
        if not self.interleaver.interleaving:
            raise RuntimeError(f"{self.path}.rotate calls the engine's rotary embedding, which runs inside a trace only")
        rotary = self.rotary_emb
        # The rotary casts and moves its cos/sin cache to what it is given, and keeps the result: what it is given
        # here is in the cache's dtype, on its device, so a call never changes what the model rotates with.
        cache = next((getattr(rotary._module, name) for name in ("cos_sin_cache", "long_short_cos_sin_cache") if hasattr(rotary._module, name)), x)
        positions = torch.as_tensor(positions, device=cache.device, dtype=torch.long)
        if positions.dim() == 1 and getattr(rotary._module, "mrope_section", None):
            positions = positions.expand(3, -1)  # M-RoPE: a text token sits at the same position on every row
        *_, tokens, width = x.shape
        rows = x.reshape(-1, tokens, width).transpose(0, 1).to(device=cache.device, dtype=cache.dtype)
        rows = rows.clone(memory_format=torch.contiguous_format).reshape(tokens, -1)  # the kernel's to rotate in place
        rotated, _ = rotary(positions, rows, rows.clone())  # a key too: the M-RoPE kernel takes both
        return rotated.reshape(tokens, -1, width).transpose(0, 1).reshape(x.shape).to(device=x.device, dtype=x.dtype)
