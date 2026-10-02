"""`SelectiveScan`: a Mamba-1 mixer's values, at the selective-scan kernel call."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from nnsight.intervention.envoy import Envoy

from jaxtyping import Float
from torch import Tensor

from .eproperty import DerivedEProperty, EProperty
from .recurrent import RecurrentMixer, _dispatch, _modeling_module, _name, kernel, needs_recurrent_routing, needs_torch_kernels

#: The layouts at the scan, tokens before channels. ``B`` and ``C`` are one ``state_dim`` vector per token,
#: shared by every channel: one group. The input, the step sizes and the read are one number per channel; the
#: decay is per channel and state dimension; the state is one ``state_dim`` vector per channel.
ScanQK = Float[Tensor, "batch seq groups state_dim"]
ScanValues = Float[Tensor, "batch seq channels"]
ScanSteps = Float[Tensor, "batch seq channels"]
ScanDecays = Float[Tensor, "batch seq channels state_dim"]
ScanState = Float[Tensor, "batch channels state_dim"]
ScanStates = Float[Tensor, "batch seq channels state_dim"]


def _decoding(envoy: Envoy) -> bool:
    """Whether this call runs the single-step decode kernel rather than the prompt's scan."""
    cls = type(envoy)
    return cls.KERNEL(envoy) == cls.RECURRENT_KERNEL


def needs_kernel_source(envoy: Envoy) -> str | None:
    """Why a value read inside the kernels' own bodies is unavailable: they are not bound to the pure-torch functions."""
    reason = needs_torch_kernels(envoy)
    if reason:
        return reason
    cls = type(envoy)
    module = _modeling_module(envoy)
    for op in (cls.CHUNK_KERNEL, cls.RECURRENT_KERNEL):
        if _dispatch(getattr(module, _name(op), None)):
            return (
                f"read inside transformers' pure-torch {_name(op)}, which this process reaches through its kernel "
                "dispatcher; call nnterp.route_kernels(model.family, 'torch') before tracing this layer"
            )
    return None


def needs_token_loop(envoy: Envoy) -> str | None:
    """Why the state after each token is unavailable: the kernels are not routed, or the scan runs mambapy's parallel scan."""
    reason = needs_recurrent_routing(envoy)
    if reason:
        return reason
    from transformers.utils.import_utils import is_mambapy_available

    if getattr(envoy._module, "use_mambapy", False) and is_mambapy_available():
        return (
            "this checkpoint sets use_mambapy and mambapy is installed, so the scan runs mambapy's parallel "
            "scan, which binds no state per token; load with use_mambapy=False to read it"
        )
    return None


def _argument(name: str):
    """``select`` for one kernel argument: its position (or keyword) in whichever kernel fires on this call."""

    def select(envoy: Envoy) -> int | str:
        return type(envoy).ARGUMENTS[name][_decoding(envoy)]

    select.__name__ = f"argument.{name}"
    return select


def _inside(ops: str) -> Any:
    """A key at a binding inside whichever kernel fires: ``source.<KERNEL>.source.<op>.output``, the op per kernel."""

    def locate(envoy: Envoy) -> str:
        cls = type(envoy)
        kernel_op = cls.KERNEL(envoy)
        return f"source.{kernel_op}.source.{getattr(cls, ops)[kernel_op == cls.RECURRENT_KERNEL]}.output"

    locate.__name__ = f"inside.{ops}"
    return locate


def _state_output_key(envoy: Envoy) -> str:
    """The state leaving the call: the scan's returned state, or the updated state the decode kernel copies into the cache."""
    cls = type(envoy)
    if _decoding(envoy):
        return f"source.{cls.RECURRENT_KERNEL}.source.{cls.STEP_OUTPUT_OP}.output"
    return f"source.{cls.CHUNK_KERNEL}.output"


class SelectiveScan(RecurrentMixer):
    """A Mamba-1 mixer (Mamba, Falcon-Mamba, Jamba's Mamba blocks): a selective state-space scan.

    Each channel of the mixer's inner width carries a state of ``state_dim``
    numbers. For each token ``t`` the mixer computes a step size per channel,
    ``dt = softplus(dt_proj(x) + dt_bias)``, and two input-dependent vectors
    ``B`` and ``C`` of ``state_dim`` numbers shared by every channel, and
    runs, per channel::

        h_t = exp(dt_t * A) * h_{t-1} + dt_t * B_t * x_t
        y_t = C_t . h_t + D * x_t

    with ``A`` a learned ``[channels, state_dim]`` matrix of negative rates.
    The output is ``y`` gated by ``silu(z)`` and projected back by
    ``out_proj``. In the shared vocabulary: ``C`` reads the state as a
    query does, ``B`` writes the input into it as a key does, ``x`` is what
    gets written (the values), ``dt`` is how strongly each token writes
    (``betas``), and ``dt * A`` is the log of how much of the state each
    token keeps (``decays``, per channel and state dimension, where a
    DeltaNet's is one number per head).

    The values are read at the kernel call the forward makes:
    ``mamba_selective_scan(x, dt, A, B, C, D=, z=, delta_bias=, ...)`` on a
    prompt, ``mamba_selective_state_update(state, x, dt, A, B, C, D, z=,
    dt_bias=, ...)`` on a decode step (one token). The kernel's tensors are
    channel-first (``x`` and ``dt`` ``[batch, channels, seq]``, ``B`` and
    ``C`` ``[batch, state_dim, seq]``; a decode step drops the sequence
    axis); each value is a view of the argument laid out tokens first, and
    an assignment is laid back out, so reads, writes and in-place edits
    reach the kernel.

    * ``attention_queries`` / ``attention_keys``: ``C`` / ``B``,
      ``[batch, seq, 1, state_dim]`` (one group).
    * ``attention_values``: ``x``, ``[batch, seq, channels]``.
    * ``betas``: ``softplus(dt + dt_bias)``, the step size, and ``decays``:
      ``A * betas``, ``[batch, seq, channels, state_dim]``; derived, read-only.
    * ``state_input``: a decode step's cached state, ``[batch, channels,
      state_dim]`` (a copy; assign to replace it), ``None`` on a prompt, whose
      scan starts from zeros.
    * ``attention_head_outputs``: ``y`` before the gate and ``out_proj``,
      ``[batch, seq, channels]``: a binding inside the kernel.
    * ``state_output``: the state after the call's last token, as the cache
      receives it: the scan's returned state on a prompt, the updated state
      the decode kernel copies into the cache.
    * ``state`` / ``states`` / ``state_after`` / ``set_state_after``: the
      state after each token, from the scan's token loop (`STATE_OP`) on a
      prompt and the single update (`STEP_STATE_OP`) on a decode step.

    The pure-torch selective scan *is* the token loop, so
    ``route_kernels(model.family, "torch")`` binds each kernel name to its
    own pure-torch function. With ``mamba_ssm`` installed the forward
    dispatches both to its compiled kernels, which have no source to read
    inside (and need CUDA): route before the first trace.
    """

    #: The call a prompt runs through: ``mamba_selective_scan(x, dt, A, B, C, D=, z=, delta_bias=, ...)``.
    CHUNK_KERNEL = "mamba_selective_scan_0"
    #: The call each decode step runs through: ``mamba_selective_state_update(state, x, dt, A, B, C, D, z=, dt_bias=, ...)``.
    RECURRENT_KERNEL = "mamba_selective_state_update_0"
    #: Inside the scan's token loop, the binding of the state after each token's update.
    STATE_OP = "ssm_state_3"
    #: Inside the decode kernel, the binding of the updated state.
    STEP_STATE_OP = "ssm_state_0"
    #: Inside the decode kernel, the updated state cast to the cache's dtype, copied into the cache.
    STEP_OUTPUT_OP = "ssm_state_to_0"
    #: Inside the prompt's and the decode step's kernel, the binding of ``y`` after the ``D`` skip and before the gate.
    READ_OPS = ("scan_output_5", "out_1")
    #: Each argument's position (or keyword) in the prompt's and the decode step's kernel.
    ARGUMENTS = {"x": (0, 1), "dt": (1, 2), "A": (2, 3), "B": (3, 4), "C": (4, 5), "dt_bias": ("delta_bias", "dt_bias")}

    # -- layout: the kernel's channel-first tensors as tokens-first views ----------

    def _tokens_first(self, value: torch.Tensor) -> torch.Tensor:
        """``[batch, X, seq]`` (a decode step: ``[batch, X]``) as a ``[batch, seq, X]`` view."""
        return value.unsqueeze(1) if _decoding(self) else value.transpose(1, 2)

    def _channels_first(self, value: torch.Tensor) -> torch.Tensor:
        """The inverse of `_tokens_first`, for a write."""
        return value[:, 0] if _decoding(self) else value.transpose(1, 2)

    def _read(self, name: str) -> Any:
        """One argument of this call's kernel, as the kernel receives it."""
        args, kwargs = getattr(self.source, type(self).KERNEL(self)).inputs
        where = _argument(name)(self)
        return kwargs.get(where) if isinstance(where, str) else args[where]

    # -- the kernel's arguments ------------------------------------------------------

    @EProperty(kernel("inputs"), select=_argument("C"), description="C, the vector each token reads the state with", unavailable=needs_torch_kernels)
    def attention_queries(self, value: torch.Tensor) -> ScanQK:
        """``C``, ``[batch, seq, 1, state_dim]``: the vector each token reads the state with, shared by every channel."""
        return self._tokens_first(value).unsqueeze(2)

    @attention_queries.postprocess
    def attention_queries(self, value: torch.Tensor) -> torch.Tensor:
        return self._channels_first(value[:, :, 0])

    @EProperty(kernel("inputs"), select=_argument("B"), description="B, the vector each token writes into the state with", unavailable=needs_torch_kernels)
    def attention_keys(self, value: torch.Tensor) -> ScanQK:
        """``B``, ``[batch, seq, 1, state_dim]``: the vector each token's input is written into the state along, shared by every channel."""
        return self._tokens_first(value).unsqueeze(2)

    @attention_keys.postprocess
    def attention_keys(self, value: torch.Tensor) -> torch.Tensor:
        return self._channels_first(value[:, :, 0])

    @EProperty(kernel("inputs"), select=_argument("x"), description="x, the input the scan writes into the state", unavailable=needs_torch_kernels)
    def attention_values(self, value: torch.Tensor) -> ScanValues:
        """``x``, ``[batch, seq, channels]``: the input each channel writes into its state, after the conv and the activation."""
        return self._tokens_first(value)

    @attention_values.postprocess
    def attention_values(self, value: torch.Tensor) -> torch.Tensor:
        return self._channels_first(value)

    def _betas(self) -> ScanSteps:
        dt, bias = self._read("dt"), self._read("dt_bias")
        if bias is not None:
            bias = bias.to(dt.dtype)
            dt = dt + (bias if _decoding(self) else bias[..., None])
        return self._tokens_first(F.softplus(dt))

    #: The step size: how strongly each token writes into each channel's state, and how far the state decays.
    betas = DerivedEProperty(
        _betas,
        description="The step size dt = softplus(dt + dt_bias), per token and channel; derived, read-only",
        unavailable=needs_torch_kernels,
    )

    def _decays(self) -> ScanDecays:
        A = self._read("A")
        return A[None, None] * self._betas()[..., None].to(A.dtype)

    #: The log decay ``dt * A``: ``exp`` of it is the factor each token keeps of the state, per channel and state dimension.
    decays = DerivedEProperty(
        _decays,
        description="The per-token log decay dt * A of the state; derived, read-only",
        unavailable=needs_torch_kernels,
    )

    @EProperty(kernel("inputs"), select=lambda envoy: 0 if _decoding(envoy) else None, description="The state entering the layer on a decode step (a copy of the cache's buffer), None on a prompt", unavailable=needs_torch_kernels)
    def state_input(self, value: Any) -> ScanState | None:
        """The state this call starts from: the cached state on a decode step, ``None`` on a prompt.

        A prompt's scan takes no state: it starts from zeros. On a decode
        step, a copy: the kernel updates the cache's buffer in place, so the
        live tensor would read as this step's output by the time the trace
        ends. Assign to replace what the step starts from.
        """
        return value.clone() if _decoding(self) else None

    @state_input.postprocess
    def state_input(self, value: torch.Tensor) -> torch.Tensor:
        if not _decoding(self):
            raise ValueError("a prompt's selective scan starts from zeros and takes no state to replace")
        return value

    # -- inside the kernels --------------------------------------------------------------

    @EProperty(_inside("READ_OPS"), description="y = C.h + D.x, the scan's read of the state before the gate and out_proj", unavailable=needs_kernel_source)
    def attention_head_outputs(self, value: torch.Tensor) -> ScanValues:
        """``y = C . h + D * x``, ``[batch, seq, channels]``: the scan's read before ``silu(z)`` and ``out_proj``."""
        return self._tokens_first(value)

    @attention_head_outputs.postprocess
    def attention_head_outputs(self, value: torch.Tensor) -> torch.Tensor:
        return self._channels_first(value)

    @EProperty(_state_output_key, select=lambda envoy: None if _decoding(envoy) else 1, description="The state leaving the layer", unavailable=needs_kernel_source)
    def state_output(self, value: torch.Tensor) -> ScanState:
        """The state after this call's last token, ``[batch, channels, state_dim]``: what the next decode step starts from.

        The scan's returned state on a prompt; on a decode step the updated
        state as the kernel copies it into the cache. Either way it is what
        the cache receives: assigning it changes what the next step starts
        from, not this call's output.
        """
        return value

    # -- the state after every token: the base's, in the scan's layout --------------------

    @EProperty(RecurrentMixer._token_state_op, description="The state after one token of the prompt; iterate it with tracer.iter; needs route_kernels(family, 'torch')", unavailable=needs_token_loop)
    def state(self, value: torch.Tensor) -> ScanState:
        """The state after a token, ``[batch, channels, state_dim]``: one occurrence per token (see `RecurrentMixer.state`)."""
        return value

    def _scan_states(self) -> ScanStates:
        return self._states()

    #: The state after each token, stacked on a sequence axis (see `RecurrentMixer.states`).
    states = DerivedEProperty(
        _scan_states,
        description="The state after every token of this call; needs route_kernels(family, 'torch')",
        unavailable=needs_token_loop,
    )
