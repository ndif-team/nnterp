"""`Attention`: a softmax-attention module, its contribution, its pattern and its interior."""

from __future__ import annotations

import inspect
import sys
from functools import partial
from typing import Any, Callable

import torch
from nnsight.intervention.envoy import Envoy
from nnsight.intervention.interleaver import Mediator

from jaxtyping import Float
from torch import Tensor

from .eproperty import DerivedEProperty, EProperty, Unavailable, tensor_parallel_mesh
from .layer import Residual
from .standard import (
    BATCH, HEAD_DIM, HEADS, KEY, KV_HEADS, QK_HEAD_DIM, QK_ROPE_HEAD_DIM, QUERY, SEQ, Standard, absent, first_tensor, in_width, module_int, rewrap, unsized,
)


def needs_eager(envoy: Envoy) -> str | None:
    """Why a value read inside the eager attention forward is unavailable, or ``None``."""
    implementation = envoy._module.config._attn_implementation
    if implementation != "eager":
        return f"read inside the eager attention forward, but this model runs {implementation!r}; load with attn_implementation='eager'"
    return None


#: The call every family on transformers' shared attention path makes:
#: ``attention_interface(module, query, key, value, attention_mask, ...)``.
INTERFACE = "attention_interface_1"

#: The call every family on transformers' shared rotary path makes, before the interface:
#: ``apply_rotary_pos_emb(query, key, cos, sin)``.
ROTARY = "apply_rotary_pos_emb_0"

#: The layouts of the interior, as transformers hands them to its attention interface: heads before tokens on
#: the queries, keys and values; ``num_kv_heads`` wide under grouped-query attention; queries and keys ``qk_head_dim``
#: deep and values ``head_dim`` deep (the same number outside latent attention).
Queries = Float[Tensor, f"{BATCH} {HEADS} {SEQ} {QK_HEAD_DIM}"]
Keys = Float[Tensor, f"{BATCH} {KV_HEADS} {SEQ} {QK_HEAD_DIM}"]
Values = Float[Tensor, f"{BATCH} {KV_HEADS} {SEQ} {HEAD_DIM}"]
#: A pattern over the tokens: the scores entering the softmax and the probabilities leaving it.
Pattern = Float[Tensor, f"{BATCH} {HEADS} {QUERY} {KEY}"]
#: Latent attention's rope key: one per position, which every head shares, ``qk_rope_head_dim`` wide (what the rotary
#: receives for the keys there, where the rest of each head's key is position-free).
SharedRopeKeys = Float[Tensor, f"{BATCH} 1 {SEQ} {QK_ROPE_HEAD_DIM}"]
#: Each head's output before they are concatenated and projected, tokens before heads.
HeadOutputs = Float[Tensor, f"{BATCH} {SEQ} {HEADS} {HEAD_DIM}"]

#: The reason a family gives for an interface value it has not mapped onto its own arithmetic.
NOT_ON_INTERFACE = (
    "The attention does its own arithmetic rather than transformers' shared "
    "attention interface; not mapped for this family yet"
)


#: Why `Attention.interface` is unavailable where no one call takes the queries, keys and values and returns the head outputs.
OWN_ARITHMETIC = (
    "the queries, keys and values are not the arguments of one call in this forward that returns the head outputs "
    "(the attention does its own arithmetic, or changes the call's output before the projection), so there is no "
    "attention function to call again"
)


def seq_first(value: torch.Tensor) -> torch.Tensor:
    """``[batch, num_heads, seq, head_dim]`` <-> ``[batch, seq, num_heads, head_dim]``, as a view.

    The standard layout of ``attention_head_outputs`` is what the shared
    interface returns, sequence before heads. A family whose own arithmetic
    keeps heads first serves a transposed view on read, so in-place edits
    still land, and transposes back on write; the transpose is its own inverse.
    """
    return value.transpose(1, 2)


def interface_reason(envoy: Envoy) -> str | None:
    return envoy.off_interface()


def rotary_reason(envoy: Envoy) -> str | None:
    return envoy.no_rotary()


def holding(envoy: Envoy, name: str) -> Envoy:
    """The nearest envoy from ``envoy`` up whose module holds a module ``name``: where the model keeps its rotary embedding."""
    host = envoy
    while host is not None and not isinstance(getattr(host._module, name, None), torch.nn.Module):
        host = host.parent
    if host is None:
        raise NotImplementedError(
            f"no {name} module above {envoy.path}; the family's {type(envoy).__name__} overrides `rotary_embedding`"
        )
    return host


def has_cache(envoy: Envoy, argument: str) -> bool:
    """Whether this call of ``envoy``'s attention receives a key/value cache: its argument ``argument`` is not ``None``.

    Read off the call's arguments once per call (`per_call`), by the first of
    the call's values that asks, so it has to ask before the forward is
    inside the call: every value read at or before the cache update asks
    (`by_cache`). Raises when a read inside the call came first.
    """
    from .recurrent import per_call

    def compute() -> bool:
        location = f"{envoy.path}.input"
        mediator = Mediator.current(location)
        # Inside the call already (its input passed, its output not yet), unless pinned to a later call by tracer.iter.
        if not mediator.iteration and mediator.occurrence(location) > mediator.occurrence(f"{envoy.path}.output"):
            raise RuntimeError(
                f"{envoy.path}: whether this call has a key/value cache is read off the call's arguments, which the "
                "forward has passed: read the attention's standard values before anything else inside it "
                "(its child modules' inputs and outputs, its own input)"
            )
        args, kwargs = Mediator.value(location)
        if argument in kwargs:
            return kwargs[argument] is not None
        names = list(inspect.signature(type(envoy._module).forward).parameters)[1:]  # without ``self``
        position = names.index(argument)
        return position < len(args) and args[position] is not None

    return per_call(envoy, "has_cache", compute)


#: The two cases `by_cache` describes, in the words the encyclopedia shows them with.
CACHE_CASES = ((True, "where the call has a key/value cache"), (False, "where it has none (use_cache=False)"))


def by_cache(
    cached: tuple[str, int | None] | Callable[[Envoy], tuple[str, int | None]],
    uncached: tuple[str, int | None] | Callable[[Envoy], tuple[str, int | None]],
    argument: str = "past_key_values",
) -> tuple[Callable[..., str], Callable[..., int | None]]:
    """An `EProperty` key and select for a value whose location depends on whether the call has a cache (`has_cache`).

    An attention that updates the cache in its own forward (``keys, values =
    cache.update(keys, values)``, BLOOM, MPT, Falcon, XGLM) computes with what
    the update returns, every position so far, where the call has a cache, and
    with the new tokens' own keys and values where it has none: ``cached`` and
    ``uncached`` are each ``(path, select)``, or a function of the host giving
    one. A value read before the update with the same location either way
    (the queries) takes ``by_cache`` too, so that reading it first does not
    leave the keys unable to ask. Outside a trace, ``locate(envoy, cache=True)``
    or ``cache=False`` gives either case's key (the encyclopedia shows both).
    """

    def choose(envoy: Envoy, cache: bool | None) -> tuple[str, int | None]:
        if cache is None:
            cache = has_cache(envoy, argument)
        case = cached if cache else uncached
        return case(envoy) if callable(case) else case

    def locate(envoy: Envoy, cache: bool | None = None) -> str:
        return choose(envoy, cache)[0]

    def select(envoy: Envoy, cache: bool | None = None) -> int | None:
        return choose(envoy, cache)[1]

    locate.__name__ = select.__name__ = "by_cache"
    locate.by_cache = select.by_cache = True
    return locate, select


def attention_call(envoy: Envoy) -> tuple[str, tuple[int, int, int]] | None:
    """The call `Attention.interface` makes again, read off the class's own values: ``(path, (q, k, v positions))``, or ``None``.

    It is the operation whose positional arguments `attention_queries`,
    `attention_keys` and `attention_values` are, provided its first return is
    `attention_head_outputs`: transformers' shared interface call on most
    families (``attention_interface(self, q, k, v, mask, ...)``), and the
    ``_attn(q, k, v, mask, head_mask)`` method of GPT-J, GPT-Neo, CodeGen and
    GPT-NeoX-Japanese. A family that maps those values onto its own arithmetic
    (BLOOM, MPT, Falcon, XGLM), or whose head outputs are read after the
    module changes the call's output (DeepSeek-V4), names no such call, and
    it is ``None`` there. Read off the keys alone, so it answers outside a trace.
    """
    cls = type(envoy)
    calls, positions = set(), []
    for value in (cls.attention_queries, cls.attention_keys, cls.attention_values):
        key = value.key if value.locate is None else None
        if not (isinstance(key, str) and key.startswith("source.") and key.endswith(".inputs") and isinstance(value.select, int)):
            return None
        calls.add(key.removesuffix(".inputs"))
        positions.append(value.select)
    heads = cls.attention_head_outputs
    if len(calls) != 1 or heads.locate is not None or heads.key != f"{next(iter(calls))}.output" or heads.select != 0:
        return None
    return next(iter(calls)), tuple(positions)


def call_reason(envoy: Envoy) -> str | None:
    """Why `Attention.interface` is unavailable, or ``None``.

    It takes the queries, keys and values and gives the head outputs and the
    pattern, so it has their reasons first (an ``sdpa`` load, a tower no image
    reaches, a call per image), then its own: no one call to make again.
    """
    cls = type(envoy)
    for name in ("attention_queries", "attention_keys", "attention_values", "attention_head_outputs", "attention_probabilities"):
        reason = getattr(cls, name).reason(envoy)
        if reason:
            return reason
    return OWN_ARITHMETIC if attention_call(envoy) is None else None


def reinvoke(self: "Attention") -> Callable[[Queries, Keys, Values], tuple[HeadOutputs, Pattern]]:
    """The attention function this forward called, bound to the rest of the live call: ``f(q, k, v) -> (head_outputs, pattern)``.

    Reading it takes the call's arguments, so it is read where the queries,
    keys and values are, in any order with them (``model.order`` ranks it
    just after them); inside ``tracer.iter`` it is that step's call. Calling
    the result runs the function the module ran (`attention_call`:
    ``eager_attention_forward`` or the family's own, with Gemma-2's logit
    softcap or GPT-OSS's sink column) on the given tensors, with every other
    argument the module passed it this time: the mask of this step, the
    scale, the dropout, the sliding window, the sinks. Make the first call
    before reading anything later in the attention (the scores, the pattern,
    the head outputs): it takes the function where the forward runs it, and
    later calls reuse it, after those reads too.

    ``q``, ``k`` and ``v`` take the layouts of `attention_queries`,
    `attention_keys` and `attention_values` (keys and values ``num_kv_heads``
    wide: the function repeats them for the groups itself), with the live
    call's query and key lengths, which its mask has. The result is the head
    outputs in the layout of `attention_head_outputs` and the pattern in that
    of `attention_probabilities`; on the served tensors they are those two
    values, bit for bit (dropout aside: in training mode it draws its own
    mask). It computes and returns; nothing reaches the model, the cache or
    the other heads until the caller writes a result::

        with model.trace(prompt):
            k = attn.attention_keys.clone()
            k[:, 0] = 0
            heads, pattern = attn.interface(attn.attention_queries, k, attn.attention_values)
            attn.attention_head_outputs[:, -1, 3] = heads[:, -1, 3]
    """
    path, positions = attention_call(self)
    call = self.get(path)
    args, kwargs = call.inputs  # this invoke's rows of the live call's arguments
    cls = type(self)
    served = (cls.attention_queries, cls.attention_keys, cls.attention_values)
    function = []  # the plain function, taken on the first call

    def interface(queries: Tensor, keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
        if not function:
            # nnsight hands a call's function over at ``.fn`` once the call is drilled into, after its
            # arguments; it hands over the plain function, not the instrumented copy it then runs, so
            # calling it again fires no operation.
            call.source
            function.append(Mediator.value(f"{call.path}.fn"))
        given = list(args)
        for position, value, tensor in zip(positions, served, (queries, keys, values)):
            # back to the argument's own layout, the way a write of the value goes
            given[position] = tensor if value._postprocess is None else value._postprocess(self, tensor)
        result = function[0](*given, **kwargs)
        return cls.attention_head_outputs._preprocess(self, result[0]), result[1]

    return interface


#: Why `Attention.value_weight` and `Attention.output_weight` are unavailable on a model split tensor-parallel.
SHARDED_WEIGHTS = (
    "the model is split tensor-parallel: each rank holds a shard (a DTensor) of every projection, so no process "
    "holds a head's rows; load it on one device to read them"
)


def projection_weight(attn: "Attention", name: str) -> Tensor:
    """The weight the attention's projection ``name`` applies, as the module holds it: what the per-head weights slice.

    The module's ``weight`` (``[out_features, in_features]`` on a ``Linear``,
    the transpose on GPT-2's ``Conv1D``), so a slice of it is a view. On a
    PEFT LoRA layer it is the weight the layer's forward applies, computed as
    ``merge()`` and ``unmerge()`` compute it, in the base weight's dtype, as a
    new tensor: the base layer's plus each active adapter's delta; the base
    weight alone once an adapter is merged into it (the forward then runs the
    base layer only); and with the adapters disabled, the base weight without
    any adapter merged into it (the forward unmerges it first).

    Raises:
        Unavailable: the block has no per-head weights (`Attention.no_head_weights`).
        NotImplementedError: an adapter whose forward is not the base layer plus a
            delta (DoRA, aLoRA, a non-LoRA tuner).
    """
    reason = attn.no_head_weights()
    if reason:
        raise Unavailable(f"{attn.path} serves no per-head weights: {reason}")
    module = attn.get(name)._module
    tuners = sys.modules.get("peft.tuners.tuners_utils")  # an adapter layer exists only once peft is imported
    if tuners is None or not isinstance(module, tuners.BaseTunerLayer):
        return module.weight
    from peft.tuners.lora import LoraLayer

    if not isinstance(module, LoraLayer) or getattr(module, "lora_variant", None) or any(getattr(module, "use_dora", {}).values()):
        raise NotImplementedError(
            f"{attn.path}.{name} is a {type(module).__name__} whose adapter is not a plain LoRA: its forward is not "
            "the base layer plus a delta, so it has no one weight to slice"
        )
    weight = module.get_base_layer().weight
    if module.disable_adapters:
        for adapter in reversed(module.merged_adapters):
            weight = weight - module.get_delta_weight(adapter).to(weight.dtype)
    elif not module.merged:
        for adapter in module.active_adapters:
            if adapter in module.lora_A:
                weight = (weight + module.get_delta_weight(adapter)).to(weight.dtype)
    return weight


class Attention(Standard):
    """A softmax-attention module: its contribution, its pattern, and its interior.

    Everything but ``attention_output`` is read inside transformers' shared
    ``eager_attention_forward``, reached through the module's
    ``attention_interface`` call (`INTERFACE`): the queries, keys and values
    it receives, the scores entering its softmax, the pattern leaving it, and
    the per-head outputs it returns. They are unavailable unless the model was
    loaded with ``attn_implementation="eager"``; `off_interface` is the one
    place that decides, so a family with another reason (GPT-2's
    ``reorder_and_upcast_attn``) overrides that method. A family whose
    attention does its own arithmetic redefines the pattern on its own op and
    marks the rest ``unavailable(NOT_ON_INTERFACE)``.

    `interface` is the function that call ran, to call again on other
    queries, keys and values with the rest of the call as it was; which call
    that is follows from where the family reads its queries, keys, values and
    head outputs (`attention_call`), so a family declares nothing for it.

    Before the interface, on the module's own forward and so under any
    ``attn_implementation``: the queries and keys the rotary embedding
    receives, read at the shared ``apply_rotary_pos_emb`` call (`ROTARY`),
    and `rotate`, which applies the block's rotary to any such tensor.
    `no_rotary` is the one place that decides a block has none.

    Its sizes (`num_heads`, `num_kv_heads`, `head_dim`, `qk_head_dim`) are
    this block's, read off the module, so they hold on a model whose blocks
    differ (Gemma-4, MiMo-V2-Flash) and outside a trace.

    The pattern is the dropout *after* the softmax, not the softmax itself:
    that is the tensor the values are mixed with on every family, after the
    cast back to the model dtype and, on a model with an attention sink
    (GPT-OSS), after the sink column is dropped.
    """

    #: Whether the softmax has an attention sink: the pattern's rows then sum to less than one, by the
    #: mass the sink took, and the scores are read just before the sink column joins them (GPT-OSS).
    SINK = False

    def off_interface(self) -> str | None:
        """Why the shared attention interface does not run on this module, or ``None``."""
        return needs_eager(self)

    def no_rotary(self) -> str | None:
        """Why this block applies no rotary embedding, or ``None``: the one place the prerope values and `rotate` ask.

        A family without rotary embeddings (learned or sinusoidal positions,
        ALiBi, none) overrides it with its reason, as does one whose blocks
        differ (a NoPE block beside rotary ones).
        """
        return None

    def no_head_weights(self) -> str | None:
        """Why this block serves no per-head weights, or ``None``: the one place `value_weight` and `output_weight` ask.

        The base slices ``v_proj`` and ``o_proj``; a family that lays its
        projections out otherwise overrides those two methods instead. A family
        whose values are not one linear map of the attention's input (latent
        attention, a norm on the values) or that has no output projection
        shared by its heads overrides this with its reason.
        """
        if tensor_parallel_mesh(self) is not None:
            return SHARDED_WEIGHTS
        return absent(self, "o_proj", where="attention")

    # -- sizes (off the module) ---------------------------------------------------
    # What this block runs with, read off the module's own attributes and, where
    # it keeps none, its projections' shapes. A family whose module spells a size
    # its own way overrides the property on its subclass.

    @property
    def head_dim(self) -> int:
        """Width of one head's values and outputs: the module's ``v_head_dim`` (latent attention, MiMo-V2-Flash), else its ``head_dim`` / ``head_size``."""
        size = module_int(self._module, "v_head_dim", "head_dim", "head_size")
        if size is None:
            raise unsized(self, "head_dim")
        return size

    @property
    def qk_head_dim(self) -> int:
        """Width of one head's queries and keys: the module's ``qk_head_dim`` (latent attention), else its ``head_dim`` / ``head_size``."""
        size = module_int(self._module, "qk_head_dim", "head_dim", "head_size")
        if size is None:
            raise unsized(self, "qk_head_dim")
        return size

    @property
    def num_heads(self) -> int:
        """Query heads: the module's ``num_heads`` / ``num_attention_heads`` / ``n_heads`` / ``n_head``, else the output projection's input width over `head_dim`."""
        size = module_int(self._module, "num_heads", "num_attention_heads", "n_heads", "n_head")
        if size is None:
            width = in_width(self._module, "o_proj", "out_proj", "dense", "c_proj", "wo")
            if width is None:
                raise unsized(self, "num_heads")
            size = width // self.head_dim
        return size

    @property
    def num_kv_heads(self) -> int:
        """Key/value heads as projected: the module's ``num_key_value_heads`` / ``num_kv_heads`` / ``kv_heads``, else `num_heads` over its ``num_key_value_groups``, else the key projection's width over `qk_head_dim`, else `num_heads`."""
        module = self._module
        size = module_int(module, "num_key_value_heads", "num_kv_heads", "kv_heads", "n_kv_heads")
        if size is not None:
            return size
        groups = module_int(module, "num_key_value_groups")
        if groups:
            return self.num_heads // groups
        width = getattr(getattr(module, "k_proj", None), "out_features", None)
        return width // self.qk_head_dim if width else self.num_heads

    @EProperty(f"source.{ROTARY}.inputs", select=0, description="The queries entering the rotary embedding", unavailable=rotary_reason)
    def attention_queries_prerope(self, value: torch.Tensor) -> Queries:
        """The queries the rotary embedding receives, ``[batch, num_heads, seq, qk_head_dim]``.

        After the query projection and any query norm, before the rotation:
        `rotate` at the positions they run at gives `attention_queries`.
        Assign to replace them; the block rotates what is written for the
        position it is written at, so a query moved from another position or
        prompt is one assignment. Served on the attention's own forward, so
        under any ``attn_implementation``.
        """
        return value

    @EProperty(f"source.{ROTARY}.inputs", select=1, description="The keys entering the rotary embedding", unavailable=rotary_reason)
    def attention_keys_prerope(self, value: torch.Tensor) -> Keys:
        """The keys the rotary embedding receives, ``[batch, num_kv_heads, seq, qk_head_dim]``.

        After the key projection and any key norm, before the rotation, the
        cache and ``repeat_kv``: the new tokens' keys only, on a decode step.
        `rotate` gives `attention_keys`; assign to replace them, rotated for
        the position they are written at.
        """
        return value

    @EProperty(f"source.{INTERFACE}.inputs", select=1, description="The queries entering attention", unavailable=interface_reason)
    def attention_queries(self, value: torch.Tensor) -> Queries:
        """The queries the attention interface receives, ``[batch, num_heads, seq, head_dim]``.

        After the query projection and, on a family with rotary embeddings,
        after they are applied (`attention_queries_prerope` is before). Assign
        to replace them. In-place edits reach the model where torch allows
        them: GPT-2's queries, keys and values are split views of one
        ``c_attn`` tensor and torch refuses to edit those in place, so assign
        there.
        """
        return value

    @EProperty(f"source.{INTERFACE}.inputs", select=2, description="The keys entering attention", unavailable=interface_reason)
    def attention_keys(self, value: torch.Tensor) -> Keys:
        """The keys the attention interface receives, ``[batch, num_kv_heads, seq, qk_head_dim]``.

        Before ``repeat_kv``, so under grouped-query attention the head axis
        is ``num_kv_heads`` wide. Assign to replace them; in-place edits reach
        the model except on GPT-2 (see `attention_queries`).
        """
        return value

    @EProperty(f"source.{INTERFACE}.inputs", select=3, description="The values entering attention", unavailable=interface_reason)
    def attention_values(self, value: torch.Tensor) -> Values:
        """The values the attention interface receives, ``[batch, num_kv_heads, seq, head_dim]``.

        Before ``repeat_kv``, like the keys. Assign to replace them; in-place
        edits reach the model except on GPT-2 (see `attention_queries`).
        """
        return value

    @EProperty(f"source.{INTERFACE}.source.nn_functional_softmax_0.input", description="The attention scores entering the softmax, masked", unavailable=interface_reason)
    def attention_scores(self, value: torch.Tensor) -> Pattern:
        """The scaled, masked scores entering the softmax, ``[batch, num_heads, query, key]``.

        ``softmax(attention_scores)`` is ``attention_probabilities`` up to the
        dtype cast. Assign to replace them; in-place edits reach the model.
        """
        return value

    @EProperty("o_proj.input", description="Every head's output side by side, as the output projection takes it", unavailable=partial(absent, name="o_proj", where="attention"))
    def attention_premix(self, value: torch.Tensor) -> HeadOutputs:
        """What the output projection receives, ``[batch, seq, num_heads, head_dim]``: the heads' outputs
        after any gate the attention applies, split by head as a view of the projection's flat input.

        The projection is ``o_proj``, the family aliasing its own spelling
        (``c_proj``, ``dense``, ``out_proj``, ``wo``) in ``RENAME``. The same
        numbers as `attention_head_outputs` on a plain attention, at a module
        boundary rather than inside the eager forward, so it is served under any
        ``attn_implementation``. In-place edits reach the model; assignment
        flattens the heads back.
        """
        return value.view(*value.shape[:-1], self.num_heads, self.head_dim)

    @attention_premix.postprocess
    def attention_premix(self, value: torch.Tensor) -> torch.Tensor:
        return value.reshape(*value.shape[:-2], -1)

    @EProperty(key="output", description="What the attention adds to the residual stream")
    def attention_output(self, value: Any) -> Residual:
        """The attention sublayer's contribution to the residual stream.

        The tensor the block adds to its input, as a tensor even when the
        module returns ``(attn_output, attn_weights)``. On a family whose
        attention adds the residual inside the module, the family's subclass
        reads the pre-residual value instead, so this always means the same
        thing. In-place edits and assignment reach the model.
        """
        return first_tensor(value)

    @attention_output.postprocess
    def attention_output(self, value: torch.Tensor) -> Any:
        return rewrap(self, value)

    @EProperty(
        f"source.{INTERFACE}.source.nn_functional_dropout_0.output",
        description="The attention pattern the values are mixed with",
        unavailable=interface_reason,
    )
    def attention_probabilities(self, value: torch.Tensor) -> Pattern:
        """The attention pattern, ``[batch, num_heads, query, key]``.

        The post-softmax probabilities as the values are mixed with them: in
        the model's dtype, and with an attention sink's column already dropped.
        Rows sum to one (to less than one on a sink model). Read it, edit it in
        place, or assign a tensor of the same shape::

            with model.trace(prompt):
                pattern = model.layers[3].self_attn.attention_probabilities.save()
                model.layers[3].self_attn.attention_probabilities[:, 5] = 0
        """
        return value

    @EProperty(f"source.{INTERFACE}.output", select=0, description="The per-head outputs before the output projection", unavailable=interface_reason)
    def attention_head_outputs(self, value: torch.Tensor) -> HeadOutputs:
        """Each head's output before they are concatenated and projected, ``[batch, seq, num_heads, head_dim]``.

        What the attention interface returns; the module reshapes it to
        ``[batch, seq, hidden_size]`` and applies the output projection to get
        ``attention_output``. Assign to replace it, or edit it in place: the
        interface returns a contiguous tensor on most families and a plain
        transposed view on GPT-2, and torch accepts in-place edits on both.
        """
        return value

    interface = DerivedEProperty(
        reinvoke,
        description="The attention function this call ran, as f(queries, keys, values) -> (head outputs, pattern), with the rest of the call's arguments",
        unavailable=call_reason,
    )

    # -- the rotary embedding -------------------------------------------------------
    # `rotate` applies the block's own rotation outside the model: the cos/sin
    # source the model computes them with and the apply function the forward
    # calls. A family whose rotary differs overrides `rotary_embedding` (the
    # source) or `apply_rotary` (the arithmetic), not `rotate`.

    def rotate(self, x: torch.Tensor, positions: Any) -> torch.Tensor:
        """``x`` rotated by this block's rotary embedding at absolute ``positions``, as the block rotates its queries and keys.

        Args:
            x: ``[..., seq, qk_head_dim]``: queries or keys the way
                `attention_queries_prerope` / `attention_keys_prerope` serve them.
            positions: Each row's absolute position, ``[seq]`` (a tensor or a
                list), or ``[batch, seq]`` for an ``x`` of ``[batch, heads, seq,
                qk_head_dim]`` whose prompts sit at different positions.

        The rotary embedding module (its rope scaling included, and the
        block's layer type where sliding and full blocks rotate differently)
        and the apply function (partial or interleaved) are the family's own,
        so ``rotate(attention_queries_prerope, positions)`` is the block's
        `attention_queries` bit for bit, and the same for the keys, where the
        block does nothing more between its rotary and its attention (a family
        that norms after the rotary, HunYuan or Llama 4, or tiles its keys,
        JetMoE, does that to the result). A rope scaling that adapts to the
        sequence (``dynamic``, ``longrope``) picks its frequencies from the
        largest position given, as the model does from the whole prompt. The
        embedding is computed by the model's rotary module wherever it lives
        and applied on ``x``'s device, so `rotate` works on a model split over
        several devices with ``device_map``. An ordinary call outside a trace;
        inside one it computes without touching the trace.

        Raises:
            Unavailable: the block applies no rotary embedding (`no_rotary`).
        """
        reason = self.no_rotary()
        if reason:
            raise Unavailable(f"{self.path}.rotate is not available: {reason}")
        positions = torch.as_tensor(positions, device=x.device)
        position_ids = positions.reshape(1, -1) if positions.dim() < 2 else positions
        embedding = self.rotary_embedding(x, position_ids)  # (cos, sin), or one tensor of complex frequencies
        embedding = tuple(e.to(x.device) for e in embedding) if isinstance(embedding, tuple) else embedding.to(x.device)
        return self.apply_rotary(x, embedding)

    def rotary_embedding(self, x: torch.Tensor, position_ids: torch.Tensor) -> Any:
        """What the block's rotary call is given for ``position_ids`` (``[batch, seq]``): ``(cos, sin)``, ``[batch, seq, width]``.

        From the rotary embedding the model computes them with, the nearest
        ``rotary_emb`` above this attention, called through its envoy, with
        this block's ``config.layer_types`` entry where its forward takes a
        ``layer_type``.
        """
        rotary = holding(self, "rotary_emb").rotary_emb
        if "layer_type" in inspect.signature(rotary._module.forward).parameters:
            layer_type = self._module.config.layer_types[self._module.layer_idx]
            return rotary(x, position_ids, layer_type)
        return rotary(x, position_ids)

    def apply_rotary(self, x: torch.Tensor, embedding: Any) -> torch.Tensor:
        """``x`` rotated by ``embedding`` with the family's ``apply_rotary_pos_emb(q, k, cos, sin)``, the function the forward calls.

        Whatever that function does is what happens: a partial rotary it
        slices itself (GPT-NeoX, Phi-3, GLM), an interleaved one (Cohere). A
        family that slices outside it, or calls another function, overrides
        this method.
        """
        cos, sin = embedding
        apply = sys.modules[type(self._module).__module__].apply_rotary_pos_emb
        heads = x.reshape(cos.shape[0], -1, *x.shape[-2:])  # [batch, heads, seq, qk_head_dim], as the forward holds it
        return apply(heads, heads, cos, sin)[0].reshape(x.shape)

    # -- the weights per head ---------------------------------------------------------
    # Read off the module, outside a trace or inside one. The base slices the
    # shared layout, head-major rows of ``v_proj`` and columns of ``o_proj``; a
    # family that fuses or transposes its projections overrides the method,
    # through `projection_weight`, which keeps the refusal and the adapters.

    def value_weight(self, kv_head: int) -> Tensor:
        """Key/value head ``kv_head``'s value weight, ``[head_dim, hidden_size]``: what maps the attention's input to that head's values.

        ``value_weight(kv) @ x``, plus that head's slice of the projection's
        bias where it has one, is ``attention_values[:, kv]`` for the
        attention's input ``x`` (`attention_norm_output`, or the stream itself
        on a post-norm block). Under grouped-query attention query head ``h``
        reads key/value head ``h // (num_heads // num_kv_heads)``. The weight's
        dtype and device; a view of the parameter, so an in-place edit is an
        edit of the model, except on a PEFT adapter (`projection_weight`).

        Raises:
            Unavailable: the block has no per-head weights (`no_head_weights`).
        """
        return projection_weight(self, "v_proj").unflatten(0, (self.num_kv_heads, self.head_dim))[kv_head]

    def output_weight(self, head: int) -> Tensor:
        """Head ``head``'s output weight, ``[hidden_size, head_dim]``: what maps its output into the residual stream.

        ``sum_j output_weight(j) @ attention_premix[:, t, j]``, plus the
        projection's bias, is what the output projection returns at ``t``. The
        weight's dtype and device; a view of the parameter except on a PEFT
        adapter (`projection_weight`).

        Raises:
            Unavailable: the block has no per-head weights (`no_head_weights`).
        """
        return projection_weight(self, "o_proj").unflatten(1, (self.num_heads, self.head_dim))[:, head]
