"""`StateSpace`: a Mamba-2 (SSD) mixer's values, at the scan kernel call."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from nnsight.intervention.interleaver import Mediator

from jaxtyping import Float
from torch import Tensor

from .eproperty import DerivedEProperty, EProperty, Unavailable, unavailable
from .linear_attention import Gates
from .recurrent import RecurrentMixer, State, States, kernel, needs_torch_kernels, per_call

#: The layouts at the scan call, tokens before heads. SSD's ``C`` and ``B`` are projected once per *group* of
#: heads (``n_groups``, which divides ``num_heads``) and live on the state's ``state_dim`` side; the values
#: ``x`` and each head's read of the state are per head, ``head_dim`` wide. The state itself is the base's
#: `State` (``batch heads key_dim value_dim``): here ``key_dim`` is ``state_dim`` and ``value_dim`` is
#: ``head_dim``.
SSDQueries = Float[Tensor, "batch seq groups state_dim"]
SSDKeys = Float[Tensor, "batch seq groups state_dim"]
SSDValues = Float[Tensor, "batch seq heads head_dim"]
SSDHeadOutputs = Float[Tensor, "batch seq heads head_dim"]


def argument(name: str):
    """A `select` function: where the scan call that fires takes ``name`` (the two kernels' call sites differ)."""

    def select(envoy: Any) -> int | str:
        return envoy._arguments_table()[name]

    select.__name__ = f"argument({name})"
    return select


def _state_output(envoy: Any) -> str:
    """The state the call leaves: the chunk scan returns it; the update writes it into the cache's buffer, so it is read inside."""
    cls = type(envoy)
    op = cls.KERNEL(envoy)
    if op == cls.RECURRENT_KERNEL:
        return f"source.{op}.source.{cls.UPDATED_STATE}.output"
    if not envoy._returns_state():
        raise Unavailable(
            f"{envoy.path}.state_output is not available: this call ran without a cache (use_cache=False), "
            "so the chunk scan returned no final state"
        )
    return f"source.{op}.output"


def _state_output_select(envoy: Any) -> int | None:
    return None if type(envoy).KERNEL(envoy) == type(envoy).RECURRENT_KERNEL else 1


def _head_outputs_select(envoy: Any) -> int | None:
    """The scan output alone: the chunk scan returns ``(output, final_state)`` when the call has a cache."""
    cls = type(envoy)
    return 0 if cls.KERNEL(envoy) == cls.CHUNK_KERNEL and envoy._returns_state() else None


def chunk_per_token(model: Any, enabled: bool = True) -> None:
    """Run every Mamba-2 mixer of ``model`` with a chunk size of 1, so the chunk scan keeps the state after every token.

    The chunk scan materializes the state at each chunk boundary
    (``new_states``, `StateSpace.states`); with ``chunk_size == 1`` every
    token is a boundary. The outputs are the same; the cost is not: the
    recurrence between chunks is quadratic in the number of chunks, so a long
    prompt runs markedly slower. Unlike `route_kernels`, which binds a
    family's kernels process-wide, this sets an attribute of each mixer module
    of this one model (``chunk_size``, which the forward reads on every
    call), so it applies from the next trace and to no other model. Call it
    on a loaded model: a lazily built model's modules are replaced when its
    weights arrive. ``enabled=False`` restores the chunk size each mixer was
    built with (``config.chunk_size``, ``mamba_chunk_size`` on Bamba, Falcon-H1
    and GraniteMoeHybrid).
    """
    mixers = model.modules(include_fn=lambda envoy: isinstance(envoy, StateSpace))
    if not mixers:
        raise ValueError(f"{type(model).__name__} has no Mamba-2 (StateSpace) mixer")
    for envoy in mixers:
        module = envoy._module
        built = module.__dict__.setdefault("_nnterp_chunk_size", module.chunk_size)
        module.chunk_size = 1 if enabled else built


def needs_per_token_chunks(envoy: Any) -> str | None:
    """Why `StateSpace.states` is unavailable: the kernels have no source to read, or the chunk scan's chunks are longer than a token."""
    reason = needs_torch_kernels(envoy)
    if reason:
        return reason
    size = envoy._module.chunk_size
    if size != 1:
        return (
            f"the chunk scan materializes the state only at chunk boundaries, every {size} tokens "
            f"(chunk_size={size}); call nnterp.chunk_per_token(model) to set every mixer's chunk_size to 1, "
            "so every token is a boundary (slower on long prompts)"
        )
    return None


def _chunk_states_at(envoy: Any) -> str:
    cls = type(envoy)
    return f"source.{cls.CHUNK_KERNEL}.source.{cls.CHUNK_STATES}.output"


@EProperty(_chunk_states_at)
def _chunk_states(self: Any, value: torch.Tensor) -> States:
    """The chunk scan's ``new_states`` after its first entry, key side first: a copy, so an in-place edit does not reach the scan."""
    return value[:, 1:].transpose(-1, -2).clone()


#: Why `StateSpace.state` is unavailable.
NO_STATE_OCCURRENCES = (
    "the chunk scan computes every token's state in one tensor per call, not one occurrence per token "
    "to walk with tracer.iter; read states, or state_after(t), after nnterp.chunk_per_token(model)"
)
#: Why `StateSpace.set_state_after` is unavailable.
NO_STATE_WRITES = (
    "the chunk scan computes every boundary state in one cumulative step from the initial state, "
    "so a state written at token t does not flow into later tokens' states; "
    "assign state_input to change where a call starts"
)


class StateSpace(RecurrentMixer):
    """A Mamba-2 (SSD) mixer (Mamba-2, Nemotron-H, Bamba, Falcon-H1, Zamba2, GraniteMoeHybrid): a selective state space.

    SSD, the state-space dual, is linear attention with a scalar decay per
    head. Per head, with the state ``h`` a ``head_dim`` by ``state_dim``
    matrix, each token does::

        h = exp(dt * A) * h + dt * x B^T
        y = h C + D * x

    so the shared names mean what they mean on a gated DeltaNet
    (`LinearAttention`):

    * ``attention_queries`` is ``C``, what reads the state;
      ``attention_keys`` is ``B``, where a token writes into it; both
      ``[batch, seq, groups, state_dim]``, one per group of heads
      (``n_groups``), shared by the heads in the group.
    * ``attention_values`` is ``x`` (``hidden_states`` at the call), what is
      written, ``[batch, seq, heads, head_dim]``.
    * ``betas`` is ``dt = softplus(dt + dt_bias)``, the write strength,
      ``[batch, seq, heads]``; ``decays`` is ``A * dt``, the log of how much of
      the state each token keeps, ``[batch, seq, heads]``, float32 and
      non-positive. Both are the kernel's ``dt`` argument seen through
      ``dt_bias``, the softplus and ``A``; an assignment is carried back into
      ``dt`` (`_dt_for`), so the kernel computes the assigned gate.
    * ``state_input`` / ``state_output``: the state entering and leaving the
      call, served as ``[batch, heads, state_dim, head_dim]`` (key side
      first, the shared `State` layout), the transpose of the cache's
      ``[batch, heads, head_dim, state_dim]``; an assignment is transposed
      back. ``state_input`` is ``None`` on a fresh prompt.
    * ``attention_head_outputs`` is ``y``, the scan's output with the ``D``
      skip, before the gated norm and ``out_proj``,
      ``[batch, seq, heads, head_dim]``.

    Everything but ``attention_output`` is read at the scan call:
    transformers' ``mamba2_chunk_scan`` on a prompt, its
    ``mamba2_selective_state_update`` on a decode step (`RecurrentMixer.KERNEL`). The two
    take their arguments in different places and the update has no sequence
    axis, so each value selects by the kernel that fires (`argument`) and a
    decode step's tensors are served with a sequence axis of 1, removed again
    on assignment. The update writes the new state into the cache's buffer
    and returns only ``y``, so a decode step's ``state_output`` is read inside
    it, at ``UPDATED_STATE``.

    The chunk scan materializes the state at every chunk boundary, in one
    tensor (``new_states``, `CHUNK_STATES`); with a chunk size of 1
    (`chunk_per_token`) every token is a boundary, and ``states`` and
    ``state_after(t)`` read the state after every token of the call (a decode
    step's is its ``state_output``). There is no per-token occurrence to walk
    with ``tracer.iter`` (``state`` is unavailable, ``STATE_OP`` is ``None``),
    and no per-token write: the scan computes every boundary state in one
    cumulative step from the initial state, so ``set_state_after`` is
    unavailable too. With ``mamba_ssm`` installed the kernels have no Python
    source; ``route_kernels(model.family, "torch")`` binds transformers'
    pure-torch ones (`RecurrentMixer`).

    A value that needs another argument of the call (``betas`` needs
    ``dt_bias``, a prompt's ``state_output`` whether the scan returns a
    state) reads the call's arguments once per call (`_arguments`), at the
    call's first need, and finds them there after the model has moved into
    the kernel. A value read after the model has moved past its location is
    an out-of-order read.
    """

    #: The call a prompt runs through: ``mamba2_chunk_scan(hidden_states, dt, A, B, C, chunk_size=, D=, dt_bias=, initial_states=, ...)``.
    CHUNK_KERNEL = "mamba2_chunk_scan_0"
    #: The call each decode step runs through: ``mamba2_selective_state_update(state, hidden_states, dt, A, B, C, D, dt_bias=, ...)``.
    RECURRENT_KERNEL = "mamba2_selective_state_update_0"
    #: Neither kernel binds the state once per token.
    STATE_OP = None
    #: Inside the chunk scan, the binding of the state at every chunk boundary: ``[batch, chunks + 1, heads, head_dim, state_dim]``, the state before each chunk and after the last.
    CHUNK_STATES = "new_states_0"
    #: Inside the update, the binding of the new state before it is copied into the cache.
    UPDATED_STATE = "ssm_states_0"
    #: Where each kernel's call site passes each argument: a position, or a keyword.
    CHUNK_ARGUMENTS = {"hidden_states": 0, "dt": 1, "A": 2, "B": 3, "C": 4, "dt_bias": "dt_bias", "state": "initial_states"}
    RECURRENT_ARGUMENTS = {"state": 0, "hidden_states": 1, "dt": 2, "A": 3, "B": 4, "C": 5, "dt_bias": "dt_bias"}

    # -- the call ------------------------------------------------------------------

    def _decoding(self) -> bool:
        return type(self).KERNEL(self) == type(self).RECURRENT_KERNEL

    def _arguments_table(self) -> dict[str, int | str]:
        return self.RECURRENT_ARGUMENTS if self._decoding() else self.CHUNK_ARGUMENTS

    def _arguments(self) -> dict[str, Any]:
        """This call's scan arguments by name, read from the model once per call, on the call's first need."""
        location = f"{getattr(self.source, type(self).KERNEL(self)).path}.input"
        args, kwargs = per_call(self, "arguments", lambda: Mediator.value(location))
        named = {name: kwargs.get(at) if isinstance(at, str) else args[at] for name, at in self._arguments_table().items()}
        named["dt_softplus"] = kwargs.get("dt_softplus", False)
        named["dt_limit"] = kwargs.get("dt_limit")
        named["return_final_states"] = kwargs.get("return_final_states", False)
        return named

    def _returns_state(self) -> bool:
        return bool(self._arguments()["return_final_states"])

    def _with_seq(self, value: torch.Tensor) -> torch.Tensor:
        """A decode step's tensor with a sequence axis of 1; a prompt's as it is."""
        return value.unsqueeze(1) if self._decoding() else value

    def _without_seq(self, value: torch.Tensor) -> torch.Tensor:
        return value.squeeze(1) if self._decoding() else value

    # -- the values at the scan call -------------------------------------------------

    @EProperty(kernel("inputs"), select=argument("C"), description="C, the queries reading the state", unavailable=needs_torch_kernels)
    def attention_queries(self, value: torch.Tensor) -> SSDQueries:
        """``C``: what each token reads the state with, ``[batch, seq, groups, state_dim]``, one per group of heads, after the conv and the activation."""
        return self._with_seq(value)

    @attention_queries.postprocess
    def attention_queries(self, value: torch.Tensor) -> torch.Tensor:
        return self._without_seq(value)

    @EProperty(kernel("inputs"), select=argument("B"), description="B, the keys writing into the state", unavailable=needs_torch_kernels)
    def attention_keys(self, value: torch.Tensor) -> SSDKeys:
        """``B``: where each token writes into the state, ``[batch, seq, groups, state_dim]``, one per group of heads."""
        return self._with_seq(value)

    @attention_keys.postprocess
    def attention_keys(self, value: torch.Tensor) -> torch.Tensor:
        return self._without_seq(value)

    @EProperty(kernel("inputs"), select=argument("hidden_states"), description="x, the values written into the state", unavailable=needs_torch_kernels)
    def attention_values(self, value: torch.Tensor) -> SSDValues:
        """``x``: what each token writes into the state, ``[batch, seq, heads, head_dim]``, after the conv and the activation."""
        return self._with_seq(value)

    @attention_values.postprocess
    def attention_values(self, value: torch.Tensor) -> torch.Tensor:
        return self._without_seq(value)

    def _gate(self, dt: torch.Tensor) -> Gates:
        """The kernel's ``dt`` argument as the write strength it computes: plus ``dt_bias``, softplus, and on a prompt clamped to ``dt_limit``."""
        args = self._arguments()
        bias = args["dt_bias"]
        if self._decoding():  # expanded over head_dim for the update: [batch, heads, head_dim], [heads, head_dim]
            dt, bias = dt[..., 0].unsqueeze(1), None if bias is None else bias[..., 0]
        if bias is not None:
            dt = dt + bias.to(dt.dtype)
        if args["dt_softplus"]:
            dt = F.softplus(dt)
        if not self._decoding() and args["dt_limit"] is not None:
            dt = torch.clamp(dt, min=args["dt_limit"][0], max=args["dt_limit"][1])
        return dt

    def _dt_for(self, betas: torch.Tensor) -> torch.Tensor:
        """The ``dt`` argument the kernel turns into ``betas``: the inverse of `_gate`, without the clamp.

        ``log(expm1(b)) - dt_bias`` when the kernel applies the softplus
        (written ``b + log(-expm1(-b))``, which does not overflow), so a
        ``betas`` of 0 is a ``dt`` of ``-inf`` and the softplus returns 0. On a
        decode step the ``[batch, 1, heads]`` value is expanded back over
        ``head_dim``, the update's ``[batch, heads, head_dim]``.
        """
        args = self._arguments()
        current, bias = args["dt"], args["dt_bias"]
        b = betas.float()
        if self._decoding():
            b, bias = b.squeeze(1), None if bias is None else bias[..., 0]
        dt = b + torch.log(-torch.expm1(-b)) if args["dt_softplus"] else b
        if bias is not None:
            dt = dt - bias.float()
        if self._decoding():
            dt = dt[..., None].expand(current.shape)
        return dt.to(current.dtype)

    def _A(self) -> torch.Tensor:
        """``A`` per head: the update takes it expanded to ``[heads, head_dim, state_dim]``."""
        A = self._arguments()["A"]
        return A[:, 0, 0] if self._decoding() else A

    #: ``dt`` after its bias, softplus and limit: how strongly each token writes into the state.
    @EProperty(kernel("inputs"), select=argument("dt"), description="dt, the per-token write strength into the state", unavailable=needs_torch_kernels)
    def betas(self, value: torch.Tensor) -> Gates:
        """``dt`` as the kernel uses it, ``[batch, seq, heads]``: ``softplus(dt + dt_bias)``, clamped to ``dt_limit`` on a prompt.

        Assignable: the value is carried back into the kernel's ``dt``
        argument (`_dt_for`). It must be positive where the kernel applies
        the softplus (0 stops the token's write and its decay); on a prompt
        the kernel clamps it to ``dt_limit`` again, so a value outside the
        limit runs clamped.
        """
        return self._gate(value)

    @betas.postprocess
    def betas(self, value: torch.Tensor) -> torch.Tensor:
        return self._dt_for(value)

    #: ``A * dt``: the log of how much of the state each token keeps.
    @EProperty(kernel("inputs"), select=argument("dt"), description="A * dt, the per-token log decay of the state", unavailable=needs_torch_kernels)
    def decays(self, value: torch.Tensor) -> Gates:
        """``A * dt``, ``[batch, seq, heads]``, float32 and non-positive: the log of how much of the state each token keeps.

        Assignable: ``decays / A`` is the ``betas`` it implies, carried back
        into ``dt`` like an assignment of ``betas``, so the token's write
        strength changes with it (they are one argument).
        """
        return self._A().float() * self._gate(value).float()

    @decays.postprocess
    def decays(self, value: torch.Tensor) -> torch.Tensor:
        return self._dt_for(value.float() / self._A().float())

    @EProperty(kernel("inputs"), select=argument("state"), description="The state entering the layer, or None at the start of a prompt (a copy of the cache's buffer)", unavailable=needs_torch_kernels)
    def state_input(self, value: Any) -> State | None:
        """The state this call starts from, key side first: ``None`` on a fresh prompt, the cached state on a decode step.

        A transposed copy: the cache hands the kernel its own buffer and the
        update overwrites it in place, so the live tensor would read as this
        step's *output* by the time the trace ends. Assign to replace what the
        step starts from.
        """
        return value if value is None else value.transpose(-1, -2).clone()

    @state_input.postprocess
    def state_input(self, value: Any) -> Any:
        return value if value is None else value.transpose(-1, -2)

    @EProperty(kernel("output"), select=_head_outputs_select, description="y, the per-head outputs before the gated norm and the output projection", unavailable=needs_torch_kernels)
    def attention_head_outputs(self, value: torch.Tensor) -> SSDHeadOutputs:
        """``y``: each head's read of the state plus the ``D`` skip, ``[batch, seq, heads, head_dim]``, before the gated norm and ``out_proj``."""
        return self._with_seq(value)

    @attention_head_outputs.postprocess
    def attention_head_outputs(self, value: torch.Tensor) -> torch.Tensor:
        return self._without_seq(value)

    @EProperty(_state_output, select=_state_output_select, description="The state leaving the layer", unavailable=needs_torch_kernels)
    def state_output(self, value: torch.Tensor) -> State:
        """The state after this call's last token, ``[batch, heads, state_dim, head_dim]``: what the next decode step starts from."""
        return value.transpose(-1, -2)

    @state_output.postprocess
    def state_output(self, value: torch.Tensor) -> torch.Tensor:
        return value.transpose(-1, -2)

    # -- the state after every token: the chunk scan's boundaries, with chunk_per_token ------------

    def _states(self) -> States:
        if self._decoding():
            return self.state_output.unsqueeze(1)
        self._arguments()  # read before the read inside the scan, for the values read after it
        return _chunk_states.__get__(self)

    #: The state after each token of this call, from the chunk scan's boundaries (``chunk_size`` 1).
    states = DerivedEProperty(
        _states,
        description="The state after every token of this call; needs nnterp.chunk_per_token(model)",
        unavailable=needs_per_token_chunks,
    )

    #: The chunk scan has no per-token occurrence of the state.
    state = unavailable(NO_STATE_OCCURRENCES)
    #: The chunk scan does not carry a state written at one token into the next.
    set_state_after = unavailable(NO_STATE_WRITES)

    def state_after(self, t: int) -> torch.Tensor:
        """The state after token ``t`` of this call, ``[batch, heads, state_dim, head_dim]``: ``states[:, t]``."""
        reason = type(self).states.reason(self)
        if reason:
            raise Unavailable(f"{self.path}.state_after is not available: {reason}")
        return self.states[:, t]
