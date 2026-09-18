"""PROTOTYPE — the core of the internals-addressing design, standalone.

Not wired into nnterp. It imports ``StandardizedTransformer`` and adds accessors
from outside, so nothing here changes nnterp's behaviour. Run ``demo.py``.

What it proves:
  * one address table, two different trees (llama-shaped, gpt2-shaped);
  * ``match_op`` resolving ``.source`` ops by pattern instead of by ``_n`` suffix;
  * availability decided before the first trace;
  * a layout descriptor that describes the *native* tensor;
  * per-head read/write as an index, bounded by the component's own head space;
  * forward-order ranks, an immediate refusal naming the right order, and a
    batch read that sorts for you;
  * a load-time check that catches a causally inert address;
  * plain objects: no descriptors, so it survives cloudpickle-by-value.

``match_op`` below is adapted from causalab
(``causalab/neural/engines/nnterp_engine/sources.py``); see the design document's
"licence" note — causalab ships no LICENCE file today, so the move needs an
explicit grant before it lands in MIT-licensed nnterp.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any, Callable, Iterable, Literal

import torch as th

from nnterp.rename_utils import RenamingError

# --------------------------------------------------------------------------- #
# 1. the matcher
# --------------------------------------------------------------------------- #

_OP_SUFFIX = re.compile(r"_\d+$")


class AddressResolutionError(RenamingError):
    """A pattern did not resolve to exactly one op in the installed forward."""


def match_op(
    pattern: str,
    names: Iterable[str],
    line_of: Callable[[str], str] | None = None,
) -> str:
    """The one op ``pattern`` names, or a refusal carrying the op inventory.

    Substring match. When several hit -- the systematic case is a variable that
    is assigned and then called, so both ops carry its name -- prefer the hits
    whose own source line *calls* the matched symbol. Anything still ambiguous
    refuses rather than guessing.
    """
    all_names = list(names)
    hits = [n for n in all_names if pattern in n]
    if len(hits) > 1 and line_of is not None:
        calls = [n for n in hits if f"{_OP_SUFFIX.sub('', n)}(" in line_of(n)]
        if calls:
            hits = calls
    if len(hits) == 1:
        return hits[0]
    what = "no op matches" if not hits else f"{len(hits)} ops match ({hits})"
    raise AddressResolutionError(
        f"pattern {pattern!r}: {what}. The installed forward names these ops: "
        f"{all_names}. A missing or ambiguous pattern usually means a "
        "transformers release moved this forward's code."
    )


def _line_of(source: Any) -> Callable[[str], str]:
    def line_of(name: str) -> str:
        op = getattr(source, name)
        text, line = getattr(op, "text", None), getattr(op, "line", None)
        if not isinstance(text, str) or not isinstance(line, int):
            return ""
        lines = text.splitlines()
        return lines[line - 1] if 0 <= line - 1 < len(lines) else ""

    return line_of


# --------------------------------------------------------------------------- #
# 2. the layout descriptor -- describes the NATIVE tensor, nothing more
# --------------------------------------------------------------------------- #

AxisKind = Literal[
    "batch",
    "position",
    "key_position",
    "head",
    "kv_head",
    "head_feature",
    "feature",
    "expert",
    "top_k",
    "state_row",
    "state_col",
]


@dataclasses.dataclass(frozen=True)
class Fused:
    """A feature axis that packs several logical tensors."""

    splits: int
    index: int
    mode: Literal["blocks", "per_head"]  # contiguous blocks, or interleaved per head


@dataclasses.dataclass(frozen=True)
class Axis:
    kind: AxisKind
    #: a "feature" axis that flattens a head structure names the head space it
    #: flattens ("head" or "kv_head"); downstream can then view it apart.
    heads: Literal["head", "kv_head"] | None = None
    fused: Fused | None = None


@dataclasses.dataclass(frozen=True)
class Layout:
    """What the tensor an accessor returns looks like, natively.

    nnterp never normalizes. This is the whole of what it says about shape, and
    it is enough for a downstream library to normalize without nnterp knowing
    what normalization means.
    """

    axes: tuple[Axis, ...]
    note: str = ""

    def index_of(self, kind: AxisKind) -> int | None:
        for i, axis in enumerate(self.axes):
            if axis.kind == kind:
                return i
        return None

    @property
    def head_axis(self) -> tuple[int, Axis] | None:
        """The axis a head index selects on: an explicit head axis, else the
        feature axis that flattens one."""
        for i, axis in enumerate(self.axes):
            if axis.kind in ("head", "kv_head"):
                return i, axis
        for i, axis in enumerate(self.axes):
            if axis.kind == "feature" and axis.heads is not None:
                return i, axis
        return None

    @property
    def head_space(self) -> Literal["head", "kv_head"] | None:
        found = self.head_axis
        if found is None:
            return None
        _, axis = found
        return axis.kind if axis.kind in ("head", "kv_head") else axis.heads

    @property
    def fused_axis(self) -> tuple[int, Fused] | None:
        for i, axis in enumerate(self.axes):
            if axis.fused is not None:
                return i, axis.fused
        return None

    def describe(self) -> str:
        parts = []
        for axis in self.axes:
            text = axis.kind
            if axis.heads:
                text += f"[{axis.heads}]"
            if axis.fused:
                text += f"[{axis.fused.mode} {axis.fused.index}/{axis.fused.splits}]"
            parts.append(text)
        return "(" + ", ".join(parts) + ")" + (f"  {self.note}" if self.note else "")


B, S, KS = Axis("batch"), Axis("position"), Axis("key_position")
H, KVH, D = Axis("head"), Axis("kv_head"), Axis("head_feature")


def feat(heads=None, fused=None):
    return Axis("feature", heads=heads, fused=fused)


# --------------------------------------------------------------------------- #
# 3. the address table
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class Address:
    """One row: where a standard internal lives on one family tree.

    ``module`` + ``op`` is the whole of "how to reach it"; ``layout`` is the
    whole of "what it looks like"; ``requires`` is the whole of "when it is
    there"; ``order`` is the whole of "when the forward computes it".
    """

    #: dotted module path under the layer envoy; "" is the layer itself.
    module: str
    #: ``.source`` op patterns from ``module.source`` inward, one drill between
    #: consecutive elements. Substring patterns, never hardcoded ``_n``.
    op: tuple[str, ...] = ()
    #: which handle of the last op (or of the module) carries the value.
    handle: Literal["output", "input", "inputs"] = "output"
    #: indices into the handle, applied in order. ``(0, 1)`` on "inputs" is
    #: positional argument 1; ``(1, "g")`` is keyword argument ``g``.
    select: tuple[int | str, ...] = ()
    #: the native tensor's axes.
    layout: Layout = dataclasses.field(default_factory=lambda: Layout(()))
    #: rank in the block's forward. Reads must be non-decreasing in
    #: ``(layer, order)``; nnsight refuses a value the forward has passed.
    order: int = 0
    #: capability flags this address needs (see CAPABILITIES).
    requires: frozenset[str] = frozenset()
    #: False when a write here reaches nothing downstream (read-only address).
    writable: bool = True
    #: False for rows only defined on softmax-attention blocks of a hybrid.
    attention_row: bool = False


_EAGER = frozenset({"attn_eager"})
_IFACE = "attention_interface"

#: the llama-shaped tree: q/k/v projections, gated MLP, softmax attention
#: through ``ALL_ATTENTION_FUNCTIONS``. Also serves mistral, qwen2/3, gemma...
LLAMA_TREE: dict[str, Address] = {
    "attention_queries_pre_rope": Address(
        "self_attn.q_proj",
        layout=Layout((B, S, feat(heads="head")), "q_proj is head-major flat"),
        order=10,
        attention_row=True,
    ),
    "attention_keys_pre_rope": Address(
        "self_attn.k_proj",
        layout=Layout((B, S, feat(heads="kv_head")), "KV space under GQA"),
        order=11,
        attention_row=True,
    ),
    "attention_values": Address(
        "self_attn.v_proj",
        layout=Layout((B, S, feat(heads="kv_head")), "KV space under GQA"),
        order=12,
        attention_row=True,
    ),
    # post-RoPE q and k are the attention function's arguments 1 and 2. k is
    # pre-repeat_kv, so still KV space. Family-agnostic: GPT-2 has no RoPE op.
    "attention_queries": Address(
        "self_attn",
        (_IFACE,),
        "inputs",
        (0, 1),
        Layout((B, H, S, D), "post-RoPE"),
        order=20,
        attention_row=True,
    ),
    "attention_keys": Address(
        "self_attn",
        (_IFACE,),
        "inputs",
        (0, 2),
        Layout((B, KVH, S, D), "post-RoPE, pre repeat_kv"),
        order=21,
        attention_row=True,
    ),
    # the softmax's input and output, addressed by the softmax CALL rather than
    # by the variable binding: `attn_weights_2` means a different tensor on the
    # llama and gpt2 copies of eager_attention_forward.
    "attention_scores": Address(
        "self_attn",
        (_IFACE, "nn_functional_softmax"),
        "inputs",
        (0, 0),
        Layout((B, H, S, KS), "post-mask softmax input"),
        order=30,
        requires=_EAGER,
        attention_row=True,
    ),
    "attention_probabilities": Address(
        "self_attn",
        (_IFACE, "nn_functional_softmax"),
        "output",
        (),
        Layout((B, H, S, KS), "rows sum to 1"),
        order=31,
        requires=_EAGER,
        attention_row=True,
    ),
    "attention_z": Address(
        "self_attn",
        (_IFACE,),
        "output",
        (0,),
        Layout((B, S, H, D), "per-head attention output, pre o_proj"),
        order=40,
        attention_row=True,
    ),
    "attention_premix": Address(
        "self_attn.o_proj",
        handle="input",
        layout=Layout((B, S, feat(heads="head")), "attention_z, head-major flat"),
        order=41,
        attention_row=True,
    ),
    "mlp_act_fn_output": Address(
        "mlp.act_fn",
        layout=Layout((B, S, feat()), "act(gate) -- the gate branch only"),
        order=60,
    ),
    "mlp_neuron_outputs": Address(
        "mlp.down_proj",
        handle="input",
        layout=Layout((B, S, feat()), "act(gate) * up"),
        order=61,
    ),
}

#: the gpt2-shaped tree: one fused ``c_attn``, ungated MLP. Same attention
#: function, so the five interface rows are shared verbatim.
_C_ATTN = Layout((B, S, feat(heads="head", fused=None)))


def _c_attn_split(index: int, heads: str) -> Layout:
    return Layout(
        (B, S, feat(heads=heads, fused=Fused(3, index, "blocks"))),
        "c_attn emits [q | k | v] as contiguous blocks",
    )


GPT2_TREE: dict[str, Address] = {
    "attention_queries_pre_rope": Address(
        "self_attn.c_attn",
        layout=_c_attn_split(0, "head"),
        order=10,
        attention_row=True,
    ),
    "attention_keys_pre_rope": Address(
        "self_attn.c_attn",
        layout=_c_attn_split(1, "head"),
        order=11,
        attention_row=True,
    ),
    "attention_values": Address(
        "self_attn.c_attn",
        layout=_c_attn_split(2, "head"),
        order=12,
        attention_row=True,
    ),
    **{
        name: LLAMA_TREE[name]
        for name in (
            "attention_queries",
            "attention_keys",
            "attention_scores",
            "attention_probabilities",
            "attention_z",
        )
    },
    "attention_premix": Address(
        "self_attn.c_proj",
        handle="input",
        layout=Layout((B, S, feat(heads="head"))),
        order=41,
        attention_row=True,
    ),
    "mlp_act_fn_output": Address(
        "mlp.act",
        layout=Layout((B, S, feat()), "act(c_fc(x)) -- no gate on this tree"),
        order=60,
    ),
    "mlp_neuron_outputs": Address(
        "mlp.c_proj",
        handle="input",
        layout=Layout((B, S, feat()), "== mlp_act_fn_output on an ungated MLP"),
        order=61,
    ),
}

#: the hybrid's linear-attention blocks (Gated DeltaNet). Design rows: these
#: resolve on a qwen3-next / qwen3.5 block, and are listed here so the table
#: format is shown carrying a third, non-attention mixer. `deltanet_state`
#: fires once per 64-token chunk, so it is NOT an Address -- see the design
#: document, section 1 (fires != "once" gets its own accessor type).
LINEAR_ATTN_ROWS: dict[str, Address] = {
    "deltanet_conv": Address(
        "linear_attn",
        ("causal_conv1d_fn",),
        layout=Layout((B, feat(), S), "channels-first"),
        order=10,
    ),
    "deltanet_queries": Address(
        "linear_attn",
        ("query_reshape",),
        layout=Layout((B, S, KVH, D), "pre repeat_interleave: key-head space"),
        order=11,
    ),
    "deltanet_keys": Address(
        "linear_attn", ("key_reshape",), layout=Layout((B, S, KVH, D)), order=12
    ),
    "deltanet_values": Address(
        "linear_attn", ("value_reshape",), layout=Layout((B, S, H, D)), order=13
    ),
    "deltanet_beta": Address(
        "linear_attn", ("b_sigmoid",), layout=Layout((B, S, H)), order=14
    ),
    "deltanet_decay": Address(
        "linear_attn",
        ("chunk_gated_delta_rule",),
        "inputs",
        (1, "g"),
        Layout((B, S, H)),
        order=15,
    ),
    "deltanet_kernel_output": Address(
        "linear_attn",
        ("chunk_gated_delta_rule",),
        "output",
        (0,),
        Layout((B, S, H, D)),
        order=16,
    ),
}

#: deliberately wrong, kept in the prototype as the negative control: the mixer
#: does return the attention weights, and this address reads them with the right
#: shape and the right values -- but the value-weighted output is already
#: computed by then, so a write reaches nothing. verify() must catch it.
INERT_CONTROL = Address(
    "self_attn",
    (_IFACE,),
    "output",
    (1,),
    Layout((B, H, S, KS)),
    order=40,
    requires=_EAGER,
    attention_row=True,
)


# --------------------------------------------------------------------------- #
# 4. availability, decided before the first trace
# --------------------------------------------------------------------------- #


def capabilities(model) -> dict[str, tuple[bool, str]]:
    """Architecture questions with architecture answers, asked once at load."""
    eager = getattr(model._module.config, "_attn_implementation", None) == "eager"
    out = {
        "attn_eager": (
            eager,
            "this component only exists under attn_implementation='eager'; the "
            "fused kernels never materialize it. Load with "
            "StandardizedTransformer(..., enable_attention_probs=True).",
        )
    }
    if model.attention_layers:
        attn = model.layers[model.attention_layers[0]].self_attn._module
        q = getattr(attn, "q_proj", None)
        heads, head_dim = model.num_heads, head_dim_of(model)
        gated = (
            q is not None
            and head_dim is not None
            and getattr(q, "out_features", None) == 2 * heads * head_dim
        )
        out["gated_attention"] = (
            gated,
            "this mixer computes no output gate: the box exists only on the "
            "gated-attention family, whose q-projection emits [q | gate] per head.",
        )
    mlp = getattr(model.layers[0], "mlp", None)
    mlp = mlp._module if mlp is not None else None
    out["moe"] = (
        mlp is not None and hasattr(mlp, "gate") and hasattr(mlp, "experts"),
        "this block's MLP is not a sparse-MoE block.",
    )
    return out


def head_dim_of(model) -> int | None:
    cfg = model._module.config
    cfg = getattr(cfg, "text_config", cfg)
    if getattr(cfg, "head_dim", None):
        return cfg.head_dim
    if model.num_heads and model.hidden_size:
        return model.hidden_size // model.num_heads
    return None


def num_kv_heads_of(model) -> int | None:
    cfg = model._module.config
    cfg = getattr(cfg, "text_config", cfg)
    for key in ("num_key_value_heads", "num_kv_heads", "n_kv_heads"):
        if getattr(cfg, key, None):
            return getattr(cfg, key)
    return model.num_heads


def pick_tree(model) -> tuple[str, dict[str, Address]]:
    """Which family tree's rows this model uses -- decided from the tree, not
    from ``model_type``: any mixer carrying ``q_proj`` is llama-shaped."""
    if not model.attention_layers:
        return "linear_attn_only", dict(LINEAR_ATTN_ROWS)
    attn = model.layers[model.attention_layers[0]].self_attn._module
    if hasattr(attn, "q_proj"):
        return "llama_tree", dict(LLAMA_TREE)
    if hasattr(attn, "c_attn"):
        return "gpt2_tree", dict(GPT2_TREE)
    raise RenamingError(
        f"no family tree matches this mixer ({type(attn).__name__}, children="
        f"{sorted(n for n, _ in attn.named_children())})"
    )


# --------------------------------------------------------------------------- #
# 5. the accessors
# --------------------------------------------------------------------------- #


class InternalsOrderError(RenamingError):
    """A read asked for a value the forward has already passed."""


class _Order:
    """Per-trace high-water mark over ``(layer, order)``."""

    def __init__(self):
        self.key = None
        self.mark = None
        self.mark_name = None
        self.seen: list[str] = []

    def _trace(self):
        """The mediator running this trace's block. Held by strong reference:
        ``id()`` alone is reused after the previous trace is collected, which
        makes a fresh trace look like a continuation of the last one."""
        from nnsight.intervention.interleaver import Mediator

        return Mediator.current("nnterp internals")

    def touch(self, rank, name, registry):
        trace = self._trace()
        if trace is not self.key:
            self.key, self.mark, self.mark_name, self.seen = trace, rank, name, [name]
            return
        if self.mark is not None and rank < self.mark:
            asked = self.seen + [name]
            raise InternalsOrderError(
                f"{name}[{rank[0]}] comes before {self.mark_name}[{self.mark[0]}] in "
                "the forward, which this trace has already read: nnsight cannot "
                "reach back to a value the model has passed.\n"
                f"  this trace asked for: {asked}\n"
                f"  forward order is:     {registry.order_hint(asked)}\n"
                "  either read them in that order, or let nnterp sort for you: "
                f"model.internals.read({rank[0]}, *{asked})"
            )
        if name not in self.seen:
            self.seen.append(name)
        self.mark, self.mark_name = max(rank, self.mark or rank), name


class InternalAccessor:
    """``model.<name>[layer]`` and ``model.<name>[layer, head]``, read and write.

    A plain object -- no descriptor anywhere -- so it pickles by value and by
    reference, and a working-tree checkout still runs remotely.
    """

    def __init__(self, registry, name: str, address: Address):
        self.registry = registry
        self.name = name
        self.address = address

    # -- reaching -----------------------------------------------------------
    def _module(self, layer: int):
        model = self.registry.model
        if self.address.attention_row and layer in model.linear_attention_layers:
            raise RenamingError(
                f"{self.name}[{layer}]: layer {layer} is a linear-attention layer; "
                f"this row is only defined on {model.attention_layers}."
            )
        target = model.layers[layer]
        for part in filter(None, self.address.module.split(".")):
            target = getattr(target, part)
        return target

    def _node(self, layer: int):
        node = self._module(layer)
        for pattern in self.address.op:
            source = node.source
            node = getattr(source, match_op(pattern, source.names, _line_of(source)))
        return node

    def _handle(self, layer: int):
        node = self._node(layer)
        value = getattr(node, self.address.handle)
        for index in self.address.select:
            value = value[index]
        return value

    # -- head slicing (a view, so writes land) -------------------------------
    def head_space(self) -> int | None:
        kind = self.address.layout.head_space
        if kind is None:
            return None
        return (
            self.registry.model.num_heads
            if kind == "head"
            else num_kv_heads_of(self.registry.model)
        )

    def _head_index(self, head: int):
        layout = self.address.layout
        found = layout.head_axis
        if found is None:
            raise RenamingError(
                f"{self.name} has no head axis: its layout is "
                f"{layout.describe()}. Indexing it by head would be a guess."
            )
        axis_i, axis = found
        space = self.head_space()
        if not 0 <= head < space:
            raise RenamingError(
                f"{self.name} names head {head}, but this component has {space} "
                f"heads ({layout.head_space} space, layout {layout.describe()}). "
                "Under GQA a query-space index over a KV-space component would "
                "silently select nothing."
            )
        if axis.kind in ("head", "kv_head"):
            return (slice(None),) * axis_i + (head,)
        # a flat feature axis packing heads: slice the head's columns
        return (slice(None),) * axis_i + (slice(None),), axis_i, space

    def _fused_slice(self, value):
        """A fused feature axis: the columns of this address's split, as a view."""
        found = self.address.layout.fused_axis
        if found is None:
            return value
        axis_i, fused = found
        width = value.shape[axis_i] // fused.splits
        if fused.mode == "blocks":
            sl = slice(fused.index * width, (fused.index + 1) * width)
            return value[(slice(None),) * axis_i + (sl,)]
        raise NotImplementedError("per-head fusion: design section 4")

    # -- read / write --------------------------------------------------------
    def __getitem__(self, key):
        layer, head = key if isinstance(key, tuple) else (key, None)
        self.registry.check_order(self.name, layer)
        value = self._fused_slice(self._handle(layer))
        if head is None:
            return value
        return self._select_head(value, head)

    def _select_head(self, value, head: int):
        layout = self.address.layout
        found = layout.head_axis
        axis_i, axis = found
        space = self.head_space()
        if not 0 <= head < space:
            raise RenamingError(
                f"{self.name} names head {head}, but this component has {space} "
                f"heads ({layout.head_space} space, layout {layout.describe()}). "
                "Under GQA a query-space index over a KV-space component selects "
                "an empty slice rather than raising."
            )
        if axis.kind in ("head", "kv_head"):
            return value[(slice(None),) * axis_i + (head,)]
        per_head = value.shape[axis_i] // space
        sl = slice(head * per_head, (head + 1) * per_head)
        return value[(slice(None),) * axis_i + (sl,)]

    def _swap(self, layer: int, new):
        """Replace the whole value, rebuilding whatever container ``select``
        indexed into. This is the path nnterp's LayerAccessor already walks for
        tuple outputs; it is needed when the native tensor is a view torch
        refuses to write in place (a function returning several views)."""
        node = self._node(layer)
        handle = self.address.handle
        select = self.address.select
        if not select:
            setattr(node, handle, new)
            return
        current = getattr(node, handle)
        if len(select) == 1:
            index = select[0]
            rebuilt = tuple(
                new if i == index else v for i, v in enumerate(current)
            )
            setattr(node, handle, rebuilt)
            return
        if len(select) == 2 and handle == "inputs":
            args, kwargs = current
            outer, inner = select
            if outer == 0:
                args = tuple(new if i == inner else v for i, v in enumerate(args))
            else:
                kwargs = {**kwargs, inner: new}
            setattr(node, handle, (args, kwargs))
            return
        raise NotImplementedError(f"swap for select={select!r}")

    def __setitem__(self, key, new):
        layer, head = key if isinstance(key, tuple) else (key, None)
        if not self.address.writable:
            raise RenamingError(f"{self.name} is a read-only address.")
        self.registry.check_order(self.name, layer)
        partial = head is not None or self.address.layout.fused_axis is not None
        native = self._handle(layer)
        if not partial:
            self._swap(layer, new)
            return
        target = self._fused_slice(native)
        if head is not None:
            target = self._select_head(target, head)
        try:
            # the view path: every slice this class hands out aliases the model's
            # own tensor, so an in-place write lands with no write-back machinery.
            target[:] = new
        except RuntimeError:
            # torch refuses in-place on a view produced by a function returning
            # several views (gpt2's attention_z is such a transpose). Edit a
            # copy of the native tensor and swap that back instead.
            edited = native.clone()
            slot = self._fused_slice(edited)
            if head is not None:
                slot = self._select_head(slot, head)
            slot[:] = new
            self._swap(layer, edited)

    def layout(self) -> Layout:
        return self.address.layout


class InternalsRegistry:
    """``model.internals`` -- the table, its availability, and the sorted read."""

    def __init__(self, model, extra: dict[str, Address] | None = None):
        self.model = model
        self.tree, rows = pick_tree(model)
        rows.update(extra or {})
        self.rows = rows
        self.caps = capabilities(model)
        self._order = _Order()
        self.accessors: dict[str, InternalAccessor] = {}
        self.unavailable: dict[str, str] = {}
        for name, address in rows.items():
            missing = [c for c in address.requires if not self.caps.get(c, (False, ""))[0]]
            if missing:
                self.unavailable[name] = " ".join(
                    self.caps.get(c, (False, f"missing capability {c}"))[1]
                    for c in missing
                )
                continue
            ok, why = self._module_present(address)
            if not ok:
                self.unavailable[name] = why
                continue
            self.accessors[name] = InternalAccessor(self, name, address)

    def _module_present(self, address: Address) -> tuple[bool, str]:
        """Walked on the ENVOY tree, which carries nnterp's standard names --
        the raw module still calls it ``attn``, the envoy calls it ``self_attn``."""
        layers = self.model.attention_layers or list(range(self.model.num_layers))
        target, walked = self.model.layers[layers[0]], []
        for part in filter(None, address.module.split(".")):
            if not hasattr(target, part):
                children = sorted(n for n, _ in target._module.named_children())
                return False, (
                    f"the block has no module {'.'.join(walked + [part])!r} "
                    f"(children of {'.'.join(walked) or 'the block'}: {children})"
                )
            target = getattr(target, part)
            walked.append(part)
        return True, ""

    # -- the user-facing surface --------------------------------------------
    @property
    def names(self) -> list[str]:
        return sorted(self.accessors)

    def __getitem__(self, name: str) -> InternalAccessor:
        if name in self.accessors:
            return self.accessors[name]
        if name in self.unavailable:
            raise RenamingError(f"{name} is not available on this model: {self.unavailable[name]}")
        raise RenamingError(
            f"{name!r} is not a standard internal. This model's tree "
            f"({self.tree}) addresses: {sorted(self.rows)}"
        )

    def status(self) -> str:
        lines = [f"tree: {self.tree}    capabilities: " + ", ".join(
            f"{k}={'yes' if v[0] else 'no'}" for k, v in sorted(self.caps.items())
        )]
        for name in sorted(self.rows):
            if name in self.accessors:
                lines.append(f"  [ok]   {name:28s} {self.rows[name].layout.describe()}")
            else:
                lines.append(f"  [--]   {name:28s} {self.unavailable[name][:70]}")
        return "\n".join(lines)

    def order_hint(self, names: Iterable[str]) -> str:
        return " then ".join(sorted(names, key=lambda n: self.rows[n].order))

    def check_order(self, name: str, layer: int):
        self._order.touch((layer, self.rows[name].order), name, self)

    def read(self, layer: int, *names: str) -> dict[str, Any]:
        """Read several internals in one trace, in forward order, whatever order
        the caller names them in. This is the spelling a user should reach for."""
        for name in names:
            self[name]  # availability, before anything is requested
        out = {}
        for name in sorted(names, key=lambda n: self.rows[n].order):
            out[name] = self[name][layer]
        return out


# --------------------------------------------------------------------------- #
# 6. the load-time / CI check
# --------------------------------------------------------------------------- #


def expected_shape(registry, address: Address, batch: int, seq: int) -> tuple:
    model = registry.model
    heads, kv, d = model.num_heads, num_kv_heads_of(model), head_dim_of(model)
    sizes = {
        "batch": batch,
        "position": seq,
        "key_position": seq,
        "head": heads,
        "kv_head": kv,
        "head_feature": d,
    }
    shape = []
    for axis in address.layout.axes:
        if axis.kind == "feature":
            if axis.heads is not None:
                # the accessor hands back this address's split of a fused
                # projection (a view), so the expected width is one split's.
                shape.append((heads if axis.heads == "head" else kv) * d)
            else:
                shape.append(None)  # an inner width the layout does not pin
        else:
            shape.append(sizes.get(axis.kind))
    return tuple(shape)


def verify(registry, prompt="The Eiffel Tower is in the city of", causal=True, names=None):
    """One row at a time: does it resolve, is the native shape what the layout
    says, and -- the one that matters -- does a write to it move the logits.

    Structural verification is one trace for all rows (cheap, load-time).
    The causal half is one trace per row (CI, recorded, not load-time).
    """
    model = registry.model
    names = list(names or registry.names)
    report = {}
    with model.trace(prompt):
        size = model.input_size.save()   # before the logits: the forward order
        base = model.logits.save()
    batch, seq = size[0], size[1]

    # -- structural: every row in one trace, in forward order ----------------
    import nnsight

    with model.trace(prompt):
        shapes = nnsight.save({})
        for name in sorted(names, key=lambda n: registry.rows[n].order):
            shapes[name] = registry[name][0].shape
    for name in names:
        want = expected_shape(registry, registry.rows[name], batch, seq)
        got = tuple(shapes[name])
        ok = len(want) == len(got) and all(w in (None, g) for w, g in zip(want, got))
        report[name] = {"shape": got, "declared": want, "shape_ok": ok}

    if not causal:
        return report
    # -- causal: one trace per row -------------------------------------------
    for name in names:
        if not registry.rows[name].writable:
            report[name]["causal"] = None
            continue
        with model.trace(prompt):
            value = registry[name][0]
            registry[name][0] = th.zeros_like(value)
            edited = model.logits.save()
        report[name]["causal"] = not th.allclose(base, edited)
    return report


# --------------------------------------------------------------------------- #
# 7. wiring (what nnterp would do in _init_standardization)
# --------------------------------------------------------------------------- #


def attach(model, extra: dict[str, Address] | None = None) -> InternalsRegistry:
    registry = InternalsRegistry(model, extra=extra)
    object.__setattr__(model, "internals", registry)
    for name, accessor in registry.accessors.items():
        if name != "attention_probabilities":  # nnterp already ships this one
            object.__setattr__(model, name, accessor)
    return registry
