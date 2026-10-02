"""`RecurrentMixer`: how a recurrent mixer's values are reached, whatever the values are.

A recurrent mixer (a gated DeltaNet, a state-space layer) runs its sequence
through a kernel function the modeling module calls: one kernel for a prompt,
another for a one-token step over a cached state (a decode step of
``generate``), picked by what the forward binds before it branches. Its values are the arguments and results of that
call. What differs between mixers is which arguments mean what; what they
share is how the call is found, which kernel names must be transformers'
pure-torch ones for there to be a call to read inside, how the family's
kernels are rebound (`route_kernels`), and, when the token-by-token kernel
materializes it, the state after every token. That shared part is
`RecurrentMixer`; a subclass (`LinearAttention`, `SelectiveScan`, `StateSpace`)
names its kernels in class constants and declares its values.
"""

from __future__ import annotations

import inspect
import sys
from contextlib import contextmanager
from typing import Any, Callable

import torch
from greenlet import getcurrent
from nnsight.intervention.envoy import Envoy
from nnsight.intervention.interleaver import Mediator
from nnsight.intervention.source import SourceEnvoy

from jaxtyping import Float
from torch import Tensor

from .eproperty import DerivedEProperty, EProperty, Unavailable
from .layer import Residual
from .standard import Standard, first_tensor, rewrap

#: The recurrent state alone, one ``key_dim`` by ``value_dim`` matrix per head, and stacked along the tokens.
State = Float[Tensor, "batch heads key_dim value_dim"]
States = Float[Tensor, "batch seq heads key_dim value_dim"]


def _name(op: str) -> str:
    """The module-level function an op calls: ``torch_chunk_gated_delta_rule_0`` -> ``torch_chunk_gated_delta_rule``."""
    return op.rsplit("_", 1)[0]


def _dispatch(bound: Any) -> dict[str, Any]:
    """The closure of transformers' kernel dispatcher (``use_kernel_func_from_hub_with_fallback``), or ``{}``.

    It holds ``torch_function``, the pure-torch kernel, and ``implementation``,
    the one the dispatcher calls: an optimized kernel when one is installed,
    ``torch_function`` otherwise.
    """
    return inspect.getclosurevars(bound).nonlocals if getattr(bound, "__closure__", None) else {}


def _torch_function(bound: Any) -> Callable:
    """The pure-torch kernel behind a module-level name: the dispatcher's ``torch_function``, or the name's own binding."""
    return _dispatch(bound).get("torch_function", bound)


def _modeling_module(envoy: Envoy):
    return sys.modules[type(envoy._module).__module__]


def _mixer(family) -> tuple[Any, type[RecurrentMixer]]:
    """The transformers modeling module a family's recurrent mixer lives in, and the mixer's envoy class."""
    from types import ModuleType

    if isinstance(family, ModuleType) and not hasattr(family, "ENVOYS"):
        # Already the modeling module: the mixer is the envoy a loaded family keys on one of its
        # classes, else a mixer class whose kernels it defines.
        def subclasses(cls):
            for sub in cls.__subclasses__():
                yield sub
                yield from subclasses(sub)

        mixers = list(subclasses(RecurrentMixer))
        for cls in mixers:
            envoys = getattr(sys.modules.get(cls.__module__), "ENVOYS", {})
            if any(envoy is cls and key.__module__ == family.__name__ for key, envoy in envoys.items()):
                return family, cls
        for cls in mixers:
            if cls.CHUNK_KERNEL and hasattr(family, _name(cls.CHUNK_KERNEL)):
                return family, cls
        raise ValueError(f"{family.__name__} defines no recurrent mixer's kernels")
    found = next(
        ((module, envoy) for module, envoy in family.ENVOYS.items() if issubclass(envoy, RecurrentMixer)), None
    )
    if found is None:
        raise ValueError(f"{family.__name__} has no recurrent mixer to route")
    module, envoy = found
    return sys.modules[module.__module__], envoy


def route_kernels(family, kernel: str = "torch") -> None:
    """Bind a family's recurrent kernels, process-wide.

    ``"torch"`` binds the prompt's and the decode step's kernel names in the
    family's modeling module to transformers' pure-torch kernels, the
    functions the dispatcher falls back to: the slower path, and the only one
    with Python source to read values inside. On a mixer whose token-by-token
    kernel materializes the state after every token (``STATE_OP`` set), the
    prompt's name is bound to that kernel, so a prompt runs through it and
    `RecurrentMixer.state`, `states`, `state_after` and `set_state_after`
    read and write the state at any position. On a gated DeltaNet that is
    the decode kernel, so both names are bound to it; on Mamba-1 the
    pure-torch selective scan is itself the token loop, and each name keeps
    its own (`RecurrentMixer.STEP_STATE_OP`). ``"default"`` restores what
    the module bound at import. Like installing a kernel, it applies to every
    model of the family; call it before tracing a layer, since a forward
    ``.source`` has already instrumented keeps the binding it was compiled
    with. ``family`` is the family module (``model.family``, or
    ``nnterp.families.qwen3_5_text``) or its modeling module.
    """
    module, mixer = _mixer(family)
    names = (_name(mixer.CHUNK_KERNEL), _name(mixer.RECURRENT_KERNEL))
    originals = module.__dict__.setdefault("_nnterp_kernels", {})
    for name in names:
        originals.setdefault(name, getattr(module, name))
    if kernel == "torch":
        bindings = {name: _torch_function(originals[name]) for name in names}
        if mixer.STATE_OP is not None:
            bindings[names[0]] = _torch_function(originals[_name(mixer._loop_kernel())])
    elif kernel == "default":
        bindings = {name: originals[name] for name in names}
    else:
        raise ValueError(f"kernel must be 'torch' or 'default', not {kernel!r}")
    for name, function in bindings.items():
        setattr(module, name, function)


def route_delta_rule(family, kernel: str = "recurrent") -> None:
    """`route_kernels` for a gated DeltaNet, in its own words.

    ``"recurrent"`` is ``route_kernels(family, "torch")``: the family's
    prompts run through transformers' token-by-token gated delta rule, which
    materializes the state after every token. ``"chunked"`` is
    ``route_kernels(family, "default")``.
    """
    routes = {"recurrent": "torch", "chunked": "default"}
    if kernel not in routes:
        raise ValueError(f"delta_rule must be 'recurrent' or 'chunked', not {kernel!r}")
    route_kernels(family, routes[kernel])


def needs_torch_kernels(envoy: Envoy) -> str | None:
    """Why a recurrent mixer's kernel values are unavailable: the kernel has no Python source to read inside."""
    cls = type(envoy)
    module = _modeling_module(envoy)
    for op in (cls.CHUNK_KERNEL, cls.RECURRENT_KERNEL):
        closure = _dispatch(getattr(module, _name(op), None))
        implementation = closure.get("implementation")
        if implementation is not None and implementation is not closure["torch_function"]:
            package = getattr(implementation, "__module__", None) or "an installed package"
            return (
                f"read inside transformers' pure-torch {_name(op)}, but this process dispatches it "
                f"to an optimized kernel ({package.split('.')[0]}) with no Python source; "
                "uninstall it, or call nnterp.route_kernels(model.family, 'torch'), to read these"
            )
    return None


def needs_recurrent_routing(envoy: Envoy) -> str | None:
    """Why the per-token state is unavailable: only the token-by-token kernel materializes it."""
    cls = type(envoy)
    if cls.STATE_OP is None:
        return "this mixer's kernels do not materialize the state per token"
    reason = needs_torch_kernels(envoy)
    if reason:
        return reason
    module = _modeling_module(envoy)
    loop = _torch_function(getattr(module, _name(cls._loop_kernel())))
    step = getattr(module, _name(cls.RECURRENT_KERNEL))
    if cls.STEP_STATE_OP and (getattr(module, _name(cls.CHUNK_KERNEL)) is not loop or step is not _torch_function(step)):
        return (
            "the state after each token is read inside the kernels' pure-torch bodies, which this process "
            "reaches through transformers' kernel dispatcher. Call nnterp.route_kernels(model.family, 'torch') "
            "before tracing this layer"
        )
    if getattr(module, _name(cls.CHUNK_KERNEL)) is not loop:
        return (
            "the state after each token is materialized only by the token-by-token kernel; "
            "the chunked kernel a prompt runs through carries it between chunks. Call "
            "nnterp.route_kernels(model.family, 'torch') before tracing this layer "
            "(slower, like attn_implementation='eager')"
        )
    return None


def per_call(envoy: Envoy, key: str, compute: Callable[[], Any]) -> Any:
    """``compute()`` once per call of the envoy's module, under ``key``.

    For something several values in one call depend on and the model serves
    once: the kernel a forward branches to, a call's arguments, its sequence
    length. The record is kept until the worker is in another call of the
    module.

    Which call that is comes from two facts nnsight keeps. A read pinned by
    ``tracer.iter`` to step k is served at the k-th occurrence of its
    location, and a mixer's kernel fires once per call, so a pinned read is
    in call k. After a step's first read the pin relaxes, and outside
    ``tracer.iter`` (or on step 0) there is none; then the call is how many
    times the module's ``.output`` has been passed, which nnsight counts for
    every location whether or not it was read: inside call c that many have
    returned, and between calls it names the one about to start.

    The records live on the worker, the greenlet running this intervention
    code. Every run of every invoke has its own (a replayed ``model.edit``
    too), so two invokes reading one mixer never share a record, a record
    never outlives its run, and the tensors in it are freed with the worker.
    """
    # The index of the module call the next read lands in (see the docstring).
    # `iteration` is the pinned step, `None` once relaxed, and 0 both for step 0
    # and for no `tracer.iter` at all, which is why 0 falls through to the count.
    mediator = Mediator.current(key)
    call = mediator.iteration or mediator.occurrence(f"{envoy.path}.output")

    # One record per (module, key) on this worker: (call, value).
    records = getcurrent().__dict__.setdefault("_nnterp_per_call", {})
    slot = (envoy.path, key)
    cached = records.get(slot)

    # Compute on the first use, or in a new call of the module. `compute()` may
    # park the worker until the model reaches what it reads; the record is
    # filed under the call decided above, the one that read lands in.
    if cached is None or cached[0] != call:
        records[slot] = cached = (call, compute())
    return cached[1]


@contextmanager
def pinned(n: int | None):
    """Pin this worker's reads to occurrence ``n`` of their location, as ``tracer.iter[n]`` does; ``None`` relaxes the pin.

    A pinned read is served at the n-th occurrence of its location and
    relaxes the pin, so one read per ``with``. The pin the worker had is
    restored on the way out.
    """
    mediator = Mediator.current("pinned")
    previous, mediator.iteration = mediator.iteration, n
    try:
        yield
    finally:
        mediator.iteration = previous


def kernel(attribute: str) -> Callable[[Envoy], str]:
    """A key at whichever kernel fires on this call: ``source.<KERNEL>.<attribute>``."""

    def locate(envoy: Envoy) -> str:
        return f"source.{type(envoy).KERNEL(envoy)}.{attribute}"

    locate.__name__ = f"kernel.{attribute}"
    return locate


class RecurrentMixer(Standard):
    """A sequence mixer with a recurrent state, read at the kernel call its forward makes.

    The mechanism every recurrent mixer shares, apart from what its values
    are. A subclass names its kernels in class constants and declares
    its values at ``kernel("inputs")`` / ``kernel("output")``:

    * `BRANCH`: the binding the forward makes before it picks a kernel,
      ``True`` when the call continues from a cached state.
    * `CHUNK_KERNEL`: the call a prompt runs through.
    * `RECURRENT_KERNEL`: the call each decode step of ``generate`` runs
      through.
    * `STATE_OP`: inside the token-by-token kernel, the binding of the state
      after each token's update; ``None`` when the mixer's kernels do not
      materialize it.
    * `STEP_STATE_OP`: set when the decode kernel is a single-step update
      rather than the token loop (Mamba-1): the binding of the new state
      inside it. The prompt's kernel is then the token loop.

    `KERNEL` names the call that fires on this call of the mixer, decided
    once per call by the forward's own test: `RECURRENT_KERNEL` for one token
    over a cached state (`BRANCH` true and a sequence axis of 1 at `SEQ_OP`),
    `CHUNK_KERNEL` otherwise, a prompt or several tokens over a cached state.
    So the same values work in a ``trace`` and at every step of
    ``tracer.iter``. The kernels have to be transformers' pure-torch ones
    for there to be a call to read inside (`needs_torch_kernels`).

    The state *after every token* of a prompt exists only in the
    token-by-token kernel: a chunked kernel carries the state between chunks.
    Like eager attention, that is the user's choice:
    ``nnterp.route_kernels(model.family, "torch")`` routes the family's
    prompts through the token-by-token kernel, and then `state`, `states`,
    `state_after` and `set_state_after` read and write the state at any
    position; without it, or on a mixer with no `STATE_OP`, reading one
    raises `Unavailable` with the reason and `support` reports it. A subclass
    whose kernels keep the state another way overrides them (`StateSpace`
    reads `states` off the chunk scan's boundaries).
    """

    #: The binding the forward makes before it branches: ``True`` when the call continues from a cached state.
    BRANCH = "use_precomputed_states_0"
    #: The call a prompt runs through.
    CHUNK_KERNEL: str | None = None
    #: The call each decode step runs through.
    RECURRENT_KERNEL: str | None = None
    #: Inside the token-by-token kernel, the binding of the state after each token's update, or ``None``.
    STATE_OP: str | None = None
    #: Inside a decode kernel that updates the state once instead of looping over tokens, the binding of the
    #: new state; ``None`` when the decode kernel is the token loop and binds `STATE_OP`. Set, the prompt's
    #: kernel is the token loop (Mamba-1's pure-torch selective scan).
    STEP_STATE_OP: str | None = None
    #: The forward's masking of its input, once per call on every mixer: ``[batch, seq, ...]``, the call's length.
    SEQ_OP = "apply_mask_to_padding_states_0"

    @staticmethod
    def KERNEL(envoy: Envoy) -> str:
        """The kernel call that fires on this call of the mixer, decided once per call (`per_call`).

        The forward's own test, ``use_precomputed_states and seq_len == 1``:
        the token-by-token update for one token over a cached state, the
        prompt's kernel for everything else. The two bindings are read in the
        order the forward makes them, which differs between mixers.
        """
        cls = type(envoy)

        def choose() -> str:
            source = envoy.source
            ops = sorted((getattr(source, cls.BRANCH), getattr(source, cls.SEQ_OP)), key=lambda op: op.line)
            served = {op.name: op.output for op in ops}
            one = served[cls.BRANCH] and served[cls.SEQ_OP].shape[1] == 1
            return cls.RECURRENT_KERNEL if one else cls.CHUNK_KERNEL

        return per_call(envoy, "kernel", choose)

    @classmethod
    def _loop_kernel(cls) -> str:
        """The kernel whose pure-torch function is the token loop: the decode kernel, unless it is a single step."""
        return cls.CHUNK_KERNEL if cls.STEP_STATE_OP else cls.RECURRENT_KERNEL

    @classmethod
    def _state_op(cls, kernel: str) -> str:
        """The binding of the state after a token inside ``kernel``."""
        return cls.STEP_STATE_OP if cls.STEP_STATE_OP and kernel == cls.RECURRENT_KERNEL else cls.STATE_OP

    @EProperty(key="output", description="What the mixer adds to the residual stream")
    def attention_output(self, value: Any) -> Residual:
        """The mixer's contribution to the residual stream, a tensor."""
        return first_tensor(value)

    @attention_output.postprocess
    def attention_output(self, value: torch.Tensor) -> Any:
        return rewrap(self, value)

    # -- the state at every token: the token-by-token kernel only -------------------

    @staticmethod
    def _token_state_op(envoy: Envoy) -> str:
        # Under `tracer.iter` the pin here is a token index, and the kernel
        # call fires once per forward: the call's kernel is decided (once per
        # call, cached) and drilled into with the pin relaxed, then the pinned
        # read that follows asks for that token's occurrence of the state op.
        cls = type(envoy)
        with pinned(None):
            kernel = cls.KERNEL(envoy)
            getattr(envoy.source, kernel).source
        return f"source.{kernel}.source.{cls._state_op(kernel)}.output"

    @EProperty(_token_state_op, description="The recurrent state after one token of the prompt; iterate it with tracer.iter; needs route_kernels(family, 'torch')", unavailable=needs_recurrent_routing)
    def state(self, value: torch.Tensor) -> State:
        """The state after a token of the prompt: one occurrence per token.

        A location inside the token-by-token kernel, so it takes nnsight's
        own iteration: ``for t in tracer.iter[:]: mix.state`` reads the state
        after every token, ``tracer.iter[4]`` the one after token 4, and an
        assignment there is a write the following tokens continue from.
        Outside any ``tracer.iter`` a read is the state after token 0. It
        lives on the prompt's kernel call: under ``generate`` walk it in an
        inner ``tracer.iter`` on step 0, and take a decode step's state from
        the call's output, one token per step. Needs ``route_kernels(family, "torch")``.
        """
        return value

    def _seq(self) -> int:
        """This call's number of tokens, read off the kernel call (the queries' sequence axis on a DeltaNet)."""
        return self.attention_queries.shape[1]

    def _token_op(self) -> tuple[SourceEnvoy, int, int]:
        """This call's state-update op, the occurrence its first token is, and its sequence length, decided once per call.

        The kernel's arguments are served once per call: reading them parks
        the worker at the call's start, which is the one moment the state
        op's occurrence count is the number of tokens *earlier* calls put
        through it. Occurrences are counted per location since the op was
        drilled into this run, so a decode step's single token is not
        occurrence 0 once an earlier step has fired the recurrent kernel's
        op. Whichever per-token read comes first in the call takes all three;
        the rest reuse them.
        """

        def compute() -> tuple[SourceEnvoy, int, int]:
            seq = self._seq()
            kernel = type(self).KERNEL(self)
            op = getattr(getattr(self.source, kernel).source, self._state_op(kernel))
            location = f"{op.path}.output"
            return op, Mediator.current(location).occurrence(location), seq

        return per_call(self, "call", compute)

    def _states(self) -> States:
        # Through whichever kernel call fires on this step, so a decode step's
        # one-token call answers too; `state` alone is the prompt's location.
        op, first, seq = self._token_op()
        states = []
        for t in range(seq):
            with pinned(first + t):
                states.append(op.output)
        return torch.stack(states, dim=1)

    #: The state after each token, stacked on a sequence axis: one occurrence of the token-by-token
    #: kernel's state update per token, so it needs ``route_kernels(family, "torch")``. The last is the
    #: state the call leaves. Read-only, a stack of copies: `set_state_after` writes one position.
    states = DerivedEProperty(
        _states,
        description="The recurrent state after every token of this call; needs route_kernels(family, 'torch')",
        unavailable=needs_recurrent_routing,
    )

    def _require_state(self, name: str) -> None:
        reason = type(self).state.reason(self)
        if reason:
            raise Unavailable(f"{self.path}.{name} is not available: {reason}")

    def state_after(self, t: int) -> torch.Tensor:
        """The state after token ``t`` of this call: ``for _ in tracer.iter[t]: mix.state`` on a prompt, as a call that also counts from the call's own first token on a decode step."""
        self._require_state("state_after")
        op, first, _ = self._token_op()
        with pinned(first + t):
            return op.output

    def set_state_after(self, t: int, value: torch.Tensor) -> None:
        """Assign `state` at token ``t``: the tokens after it continue from ``value``.

        A write inside the prompt. Reads follow the forward: in the same
        trace, read positions before ``t`` before the write and positions
        after it afterwards; `states` (every position) only before it.
        """
        self._require_state("set_state_after")
        op, first, _ = self._token_op()
        with pinned(first + t):
            op.output = value
