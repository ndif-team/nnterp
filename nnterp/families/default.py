"""The best-effort family, for a ``model_type`` no family module covers.

`nnterp.families.lookup` falls back to this module, with a warning, when the
checkpoint's ``model_type`` has neither a module here nor a registered family.
It covers no ``MODEL_TYPES``: it is never matched by name.

``RENAME`` lists the spellings the shipped families use for the containers
(``model``, ``transformer``, ``gpt_neox``, ``model.decoder``, ``backbone``,
a multimodal ``model.language_model`` or ``language_model.model``), the
embedding, the blocks, the final norm, the head, and a block's attention, MLP
and norms. A key that does not resolve on a checkpoint is skipped, so one dict
serves every architecture. ``ENVOYS`` is keyed on the *standard* names:
nnsight matches a string key against a module's alias paths as well as its
native one, so ``"self_attn"`` reaches GPT-2's ``attn`` once ``RENAME`` has
aliased it. The blocks have no alias path of their own (``layers`` is the
container), so the container's envoy, `Layers`, wraps each of its children in
`Layer`.

What the default cannot know it reports rather than guesses: `check` runs at
load, confirms the root has what it needs (``embed_tokens``, ``layers``,
``norm``, ``lm_head``) and that the blocks pass a ``[batch, seq, hidden]``
stream on a shape-only scan, and raises `UnsupportedFamily` otherwise. A
sublayer whose output is not what the block adds to the stream (it takes the
residual itself, or the block norms or scales its output before adding it)
has its contribution reported unavailable; so is the attention interior when
the forward does not call transformers' shared attention interface. None of
the mixture, recurrent-mixer or per-family values exist here: those need a
family module (docs/extending/adding-a-family.md).
"""

from __future__ import annotations

import ast
import functools
import inspect
import sys
import textwrap
import warnings
from typing import TYPE_CHECKING, Any

import torch
from nnsight.intervention.envoy import Envoy
from nnsight.intervention.source import SourceNotAvailable

from . import UnsupportedFamily
from ..components import (
    INTERFACE, Attention, EProperty, HeadOutputs, Layer, Mlp, Pattern, Residual, Standard, first_tensor, needs_eager, rewrap,
)

if TYPE_CHECKING:
    from ..standardized import StandardizedTransformer

MODEL_TYPES = ()

#: The containers the shipped families keep their modules in, and what each root module is spelled there.
CONTAINERS = ("model", "transformer", "gpt_neox", "model.decoder", "backbone", "model.language_model", "language_model.model")
EMBEDDINGS = ("embed_tokens", "wte", "word_embeddings", "embed_in", "embeddings")
BLOCKS = ("layers", "h", "blocks")
NORMS = ("norm", "final_layernorm", "final_layer_norm", "ln_f", "norm_f", "layer_norm", "final_norm", "out_norm", "ln_out")

RENAME = {
    **{f"{container}.{name}": "embed_tokens" for container in CONTAINERS for name in EMBEDDINGS},
    **{f"{container}.{name}": "layers" for container in CONTAINERS for name in BLOCKS},
    **{f"{container}.{name}": "norm" for container in CONTAINERS for name in NORMS},
    "embed_out": "lm_head",
    "language_model.lm_head": "lm_head",
    # a block's modules: single-component keys bind wherever they resolve
    "attn": "self_attn",
    "attention": "self_attn",
    "self_attention": "self_attn",
    "norm_attn_norm.attn": "self_attn",
    "feed_forward": "mlp",
    "ffn": "mlp",
    "block_sparse_moe": "mlp",
    "ln_1": "input_layernorm",
    "ln_2": "post_attention_layernorm",
    "norm_1": "input_layernorm",
    "norm_2": "post_attention_layernorm",
    "self_attn_layer_norm": "input_layernorm",
}


# -- what the static checks read ------------------------------------------------------

def _takes_residual(module: torch.nn.Module) -> str | None:
    """The forward's parameter that looks like the residual stream, or ``None``.

    A sublayer that takes the residual (BLOOM's attention and MLP, MPT's MLP)
    adds it inside, so its output is the stream, not its contribution.
    """
    try:
        parameters = inspect.signature(type(module).forward).parameters
    except (TypeError, ValueError):
        return None
    return next((name for name in parameters if "residual" in name), None)


#: Tensor methods that only change the layout, not the values.
LAYOUT = frozenset({"reshape", "view", "contiguous", "transpose", "flatten", "permute", "to"})
#: Attributes that read a tensor's metadata, not its values: ``x.to(attn_output.dtype)`` does not use ``attn_output``.
METADATA = frozenset({"dtype", "shape", "device", "ndim"})


def _uses(node: ast.AST, name: str) -> bool:
    """Whether ``node`` reads the variable's values (not just its dtype or shape)."""
    if isinstance(node, ast.Attribute) and node.attr in METADATA:
        return False
    if isinstance(node, ast.Name):
        return node.id == name and isinstance(node.ctx, ast.Load)
    return any(_uses(child, name) for child in ast.iter_child_nodes(node))


def _on_self(expr: ast.expr) -> bool:
    """Whether a call's function is reached from ``self`` (``self.o_proj``, ``self.experts.reduce``)."""
    func = expr.func if isinstance(expr, ast.Call) else expr
    while isinstance(func, ast.Attribute):
        func = func.value
    return isinstance(func, ast.Name) and func.id == "self"


def _callee(expr: ast.expr) -> str | None:
    """The called name's last component: ``F.dropout(...)`` -> ``dropout``, ``self.ln(...)`` -> ``ln``."""
    if not isinstance(expr, ast.Call):
        return None
    func = expr.func
    return func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None


def _self_call(expr: ast.expr) -> str | None:
    """``self.<name>(...)`` -> ``name``."""
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and isinstance(expr.func.value, ast.Name):
        if expr.func.value.id == "self":
            return expr.func.attr
    return None


def _passes(expr: ast.expr, name: str, dropouts: frozenset[str]) -> bool:
    """Whether ``expr`` is the variable as is, through a dropout (the identity in eval) or a layout method: ``x``, ``self.drop(x)``, ``F.dropout(x, ...)``, ``x.view(...)``."""
    if isinstance(expr, ast.Name):
        return expr.id == name
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and expr.func.attr in LAYOUT:
        return _passes(expr.func.value, name, dropouts)  # `x.view(residual.shape)`
    dropout = _self_call(expr) in dropouts or _callee(expr) == "dropout"
    return dropout and bool(expr.args) and _passes(expr.args[0], name, dropouts)


def _terms(expr: ast.expr) -> list[ast.expr]:
    """The terms of a sum (``a + b + c``; ``dropout_add(x, residual, ...)`` is ``x + residual``); a non-sum is its own one term."""
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        return _terms(expr.left) + _terms(expr.right)
    if _callee(expr) == "dropout_add" and len(expr.args) >= 2:
        return list(expr.args[:2])
    return [expr]


@functools.lru_cache(maxsize=None)
def _block_handling(block_type: type, child: str, dropouts: frozenset[str], fused: bool = False) -> str | None:
    """What the block's forward does to ``self.<child>(...)``'s result before adding it to the stream, or ``None`` when it adds it as is.

    Reads the forward's statements in order. The sublayer's result (the first
    element of a tuple target) is followed through dropouts (the identity in
    eval) and tuple unpacking; its first other use decides. A sum with it as a
    plain term is the contribution as is, unless every other term is another
    module's call (a mixture's output plus ``self.shared_mlp(...)``). A call on
    another of the block's modules (a post-sublayer norm), an in-place change
    or any other arithmetic (a scale) is not. A forward that does not assign the call in a
    way this reads is given the benefit of the doubt. With ``fused``, a module
    call taking it and another argument (``self.post_attention_layernorm(x,
    residual)``) is the add, fused into the next norm as vLLM's blocks do, and
    so is returning it first in a pair, for the next block's norm to add.
    """
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(block_type.forward)))
    except (OSError, TypeError, SyntaxError):
        return None
    function = tree.body[0]
    if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    statements = sorted(
        (node for node in ast.walk(function) if isinstance(node, (ast.Assign, ast.AugAssign, ast.Return, ast.Expr))),
        key=lambda node: (node.lineno, node.col_offset),
    )
    variable = None
    for node in statements:
        value = getattr(node, "value", None)
        if variable is not None:
            if fused and isinstance(node, ast.Return) and isinstance(value, ast.Tuple) and value.elts and _passes(value.elts[0], variable, dropouts):
                return None  # handed on as the hidden half of the fused pair, which the next block's norm adds
            if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == variable:
                return f"the block changes it in place (`{ast.unparse(node)}`), so a read would hold the result"
            if value is not None and _uses(value, variable):
                if isinstance(node, ast.Assign) and _self_call(value) == child:
                    continue  # the same call on another branch (`if ...: x = self.mlp(x) else: x = self.mlp(x, ...)`)
                if isinstance(node, ast.Assign) and _passes(value, variable, dropouts):
                    target = node.targets[0]
                    if isinstance(target, ast.Tuple) and target.elts:  # `hidden_states, _ = hidden_states`
                        target = target.elts[0]
                    if isinstance(target, ast.Name):
                        variable = target.id
                        continue
                terms = _terms(value)
                own = [term for term in terms if _passes(term, variable, dropouts)]
                if own and len(terms) > 1:
                    others = [term for term in terms if not any(term is mine for mine in own)]
                    if all(_self_call(term) for term in others):
                        return f"the block sums it with another module's output (`{ast.unparse(value)}`) before adding it to the stream"
                    return None
                called = _self_call(value)
                if called is not None and fused and len(value.args) >= 2 and _passes(value.args[0], variable, dropouts):
                    return None
                if called is not None:
                    return f"the block passes it through `self.{called}` before adding it to the stream"
                return f"the block computes `{ast.unparse(value)}` from it"
        if variable is None and isinstance(node, ast.Assign) and _self_call(value) == child:
            target = node.targets[0]
            if isinstance(target, ast.Tuple) and target.elts:
                target = target.elts[0]
            if not isinstance(target, ast.Name):
                return None
            variable = target.id
    return None


@functools.lru_cache(maxsize=None)
def _after_interface(attention_type: type, call: str = "attention_interface") -> str | None:
    """What the attention's forward computes from the interface's output before projecting it, or ``None`` when only the layout changes.

    Follows the variable the ``attention_interface(...)`` call (or ``call``,
    vLLM's ``self.attn(...)``) is assigned to;
    layout methods, arithmetic (an output gate) and the module's own
    projections keep it the head outputs; a free function applied to it (a
    rotation) does not.
    """
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(attention_type.forward)))
    except (OSError, TypeError, SyntaxError):
        return None
    variable = None
    for node in sorted((n for n in ast.walk(tree) if isinstance(n, ast.Assign)), key=lambda n: (n.lineno, n.col_offset)):
        if variable is None:
            if _callee(node.value) == call:
                target = node.targets[0]
                target = target.elts[0] if isinstance(target, ast.Tuple) and target.elts else target
                variable = target.id if isinstance(target, ast.Name) else None
                if variable is None:
                    return None
            continue
        if any(isinstance(call, ast.Call) and _on_self(call) and any(_uses(arg, variable) for arg in call.args) for call in ast.walk(node.value)):
            return None  # into the module's own projection
        for call in ast.walk(node.value):
            if isinstance(call, ast.Call) and any(_uses(arg, variable) for arg in call.args):
                is_method = isinstance(call.func, ast.Attribute) and _uses(call.func.value, variable)
                if not (is_method and call.func.attr in LAYOUT):
                    return f"the forward computes `{ast.unparse(node.value)}` from the interface's output"
        if isinstance(node.targets[0], ast.Name) and _uses(node.value, variable):
            variable = node.targets[0].id
    return None


def contribution_reason(module: torch.nn.Module, holder: torch.nn.Module, name: str, fused: bool = False) -> str | None:
    """Why ``module`` (``holder.<name>``) does not output what its block adds to the stream, or ``None`` as far as the default can tell."""
    residual = _takes_residual(module)
    if residual is not None:
        return (
            f"{type(module).__name__}.forward takes `{residual}`, so it likely adds the residual itself and its "
            "output is the stream, not the contribution; a family module points this value at the right place"
        )
    dropouts = frozenset(child for child, sub in holder.named_children() if isinstance(sub, torch.nn.Dropout))
    handling = _block_handling(type(holder), name, dropouts, fused)
    if handling is not None:
        return f"not what the block adds to the stream: {handling}; a family module points this value at the right place"
    return None


def _eager_reason(envoy: "Attention", op: str) -> str | None:
    """Why a value read at ``nn.functional.<op>`` inside the eager attention forward is unavailable, or ``None``.

    The interface's eager forward is the modeling module's
    ``eager_attention_forward``; one that does its own softmax or dropout
    (Granite's sliding-window variant) has no such op to read.
    """
    reason = envoy.off_interface()
    if reason is not None:
        return reason
    eager = getattr(sys.modules.get(type(envoy._module).__module__), "eager_attention_forward", None)
    try:
        source = inspect.getsource(eager) if eager is not None else None
    except (OSError, TypeError):
        source = None
    if source is not None and f"nn.functional.{op}(" not in source:
        return f"this model's eager_attention_forward makes no nn.functional.{op} call, where the default family reads this value; a family module maps it"
    return None


def _head_outputs_reason(envoy: "Attention") -> str | None:
    """Why the interface's output is not the head outputs the projection takes, or ``None``."""
    reason = envoy.off_interface()
    if reason is not None:
        return reason
    after = _after_interface(type(envoy._module))
    return f"{after}, so it is not what the output projection takes; a family module maps it" if after else None


def _contribution(envoy: Standard) -> str | None:
    """The reason `check` recorded on this sublayer, if any."""
    return envoy.__dict__.get("_default_reason")


# -- the envoys ----------------------------------------------------------------------

def missing_modules(block: Envoy, hosts: dict[str, type]) -> dict[str, str]:
    """``{"<module>.<value>": reason}`` for each of ``hosts`` (standard name -> its envoy class) the block has no `Standard` child under."""
    missing = {}
    for name, envoy_class in hosts.items():
        if not isinstance(block.__dict__.get(name), Standard):
            reason = f"no {name} module found under the names the default family knows; a family module names it"
            missing.update((f"{name}.{value}", reason) for value in envoy_class.values())
    return missing


class Layer(Layer):
    """A decoder block as the default finds it.

    Whether it returns a tuple is read off the load-time scan (`check`), so
    `skip_with` packs the stream the way the block does. A block with no
    module the default recognizes as its attention or MLP lists that
    module's values as unavailable, so `support` shows what was not found.
    """

    def support(self) -> dict[str, str | None]:
        return {**super().support(), **missing_modules(self, {"self_attn": Attention, "mlp": Mlp})}


class Layers(Envoy):
    """The block container: wraps each block in `Layer`.

    A block has no alias path for a string key to match (``layers`` names the
    container, ``layers.0`` one index), so the container picks its children's
    class. A key that matches a block on its own (a type key, a native path)
    still wins.
    """

    def _resolve_envoy_class(self, module: torch.nn.Module, path: str) -> type[Envoy]:
        envoy_class = super()._resolve_envoy_class(module, path)
        return Layer if envoy_class is Envoy else envoy_class


class Attention(Attention):
    """An attention module as the default finds it.

    Its interior is read on transformers' shared attention interface, as the
    base does; on a forward that makes no such call the interior is
    unavailable. ``attention_output`` is the module's output, unavailable when
    the module or its block suggests that is not the contribution
    (`contribution_reason`, recorded by `check`).
    """

    def off_interface(self) -> str | None:
        try:
            names = self.source.names
        except SourceNotAvailable:
            names = ()
        if INTERFACE not in names:
            return (
                f"{type(self._module).__name__}.forward makes no `{INTERFACE.rsplit('_', 1)[0]}` call (transformers' "
                "shared attention interface), which the default family reads the interior on; a family module maps it"
            )
        return needs_eager(self) if hasattr(self._module, "config") else None

    @EProperty(Attention.attention_scores.key, description=Attention.attention_scores.description, unavailable=functools.partial(_eager_reason, op="softmax"))
    def attention_scores(self, value: torch.Tensor) -> Pattern:
        return value

    @EProperty(Attention.attention_probabilities.key, description=Attention.attention_probabilities.description, unavailable=functools.partial(_eager_reason, op="dropout"))
    def attention_probabilities(self, value: torch.Tensor) -> Pattern:
        return value

    @EProperty(Attention.attention_head_outputs.key, select=0, description=Attention.attention_head_outputs.description, unavailable=_head_outputs_reason)
    def attention_head_outputs(self, value: torch.Tensor) -> HeadOutputs:
        return value

    @EProperty(key="output", description=Attention.attention_output.description, unavailable=_contribution)
    def attention_output(self, value: Any) -> Residual:
        return first_tensor(value)

    @attention_output.postprocess
    def attention_output(self, value: torch.Tensor) -> Any:
        return rewrap(self, value)


class Mlp(Mlp):
    """A feed-forward module as the default finds it: ``mlp_output`` is its output (the first tensor of a mixture's tuple), unavailable as `Attention`'s ``attention_output`` is."""

    @EProperty(key="output", description=Mlp.mlp_output.description, unavailable=_contribution)
    def mlp_output(self, value: Any) -> Residual:
        return first_tensor(value)

    @mlp_output.postprocess
    def mlp_output(self, value: torch.Tensor) -> Any:
        return rewrap(self, value)


#: Keyed on the standard names (nnsight matches a string key on alias paths too).
ENVOYS = {"layers": Layers, "self_attn": Attention, "mlp": Mlp}


# -- the load-time check -------------------------------------------------------------

#: What the root needs: each is read by a root value or method (`logits` through `lm_head`, `project_on_vocab`, `skip_layers`, ...).
REQUIRED = ("embed_tokens", "layers", "norm", "lm_head")


def _guess(model: "StandardizedTransformer", missing: list[str]) -> dict[str, str]:
    """A ``rename`` for the missing root modules, read off the module tree, for the error message.

    The longest module list is the blocks, the first embedding the token
    embedding, a norm beside the blocks the final norm, a linear on the root
    as wide as the vocabulary the head.
    """
    modules = dict(model._module.named_modules())
    lists = [(path, module) for path, module in modules.items() if isinstance(module, torch.nn.ModuleList) and path]
    blocks = max(lists, key=lambda item: len(item[1]))[0] if lists else None
    embeddings = [path for path, module in modules.items() if isinstance(module, torch.nn.Embedding)]
    vocab = modules[embeddings[0]].num_embeddings if embeddings else None
    guess = {
        "layers": blocks,
        "embed_tokens": embeddings[0] if embeddings else None,
        "norm": next(
            (path for path in modules if blocks and path.rpartition(".")[0] == blocks.rpartition(".")[0]
             and "norm" in type(modules[path]).__name__.lower()),
            None,
        ),
        "lm_head": next(
            (name for name, module in model._module.named_children()
             if isinstance(module, torch.nn.Linear) and module.out_features == vocab),
            None,
        ),
    }
    return {guess[name]: name for name in missing if guess[name]}


def _refuse(model: "StandardizedTransformer", what: str, rename: dict[str, str] | None = None) -> UnsupportedFamily:
    architecture = type(model._module).__name__
    hint = f"; going by the module tree, perhaps rename={rename}" if rename else ""
    return UnsupportedFamily(
        f"the default family cannot standardize {architecture}: {what}. Name the native modules with "
        f"rename={{'<native path>': '<standard name>'}} at load{hint}, or add a family module named after "
        f"the model_type (docs/extending/adding-a-family.md)."
    )


def check_names(model: Any, layer: type, required: tuple[str, ...] = REQUIRED, fused: bool = False) -> dict[tuple[str, type], list[Standard]]:
    """The part of `check` that reads no activation, shared with vLLM's default.

    Raises `UnsupportedFamily` when the root lacks one of ``required`` or no
    block is a ``layer``. Records on each block's attention and MLP the
    `contribution_reason` when there is one. Returns the sublayers by
    (standard name, module type).
    """
    missing = [name for name in required if not isinstance(model.__dict__.get(name), Envoy)]
    if not missing and not any(isinstance(block, layer) for block in model.layers):
        missing = ["layers"]
    if missing:
        raise _refuse(model, f"found no {', '.join(missing)} under any name it knows", _guess(model, missing))

    root = model._module
    prefix = f"{model.path}."
    sublayers: dict[tuple[str, type], list[Standard]] = {}  # (standard name, module type) -> its envoys
    for block in model.layers:
        for name in ("self_attn", "mlp"):
            envoy = block.__dict__.get(name)
            if not isinstance(envoy, Standard):
                continue
            holder_path, _, native = envoy.path.removeprefix(prefix).rpartition(".")
            holder = root.get_submodule(holder_path)
            reason = contribution_reason(envoy._module, holder, native, fused)
            if reason is not None:
                envoy._default_reason = reason
            sublayers.setdefault((name, type(envoy._module)), []).append(envoy)
    return sublayers


def check(model: "StandardizedTransformer") -> None:
    """Confirm the default's guess on a freshly built model; called by `StandardizedTransformer` at load.

    Raises `UnsupportedFamily` when the root lacks a module it needs or the
    blocks do not pass a ``[batch, seq, hidden]`` stream on a shape-only scan
    (fake tensors, so nothing loads or computes). Records on each block's
    attention and MLP why its output is not the contribution, when the forward
    or the scan says so, so `support` and the reads report it.
    """
    sublayers = check_names(model, Layer)
    ids = torch.zeros(1, 3, dtype=torch.long)
    first, last = model.layers[0], model.layers[-1]
    seen: dict[str, Any] = {}
    try:
        with model.scan(ids):
            seen["input"] = first.input.shape
            seen["tuple"] = isinstance(first.output, tuple)
            seen["first"] = first_tensor(first.output).shape
            seen["last"] = first_tensor(last.output).shape
            seen["logits"] = model.logits.shape
    except Exception as error:
        # Fake tensors do not run everything (grouped expert matmuls, CUDA-only kernels, data-dependent shapes):
        # a scan that cannot run says nothing about the names.
        warnings.warn(
            f"the default family's shape check could not run on {type(model._module).__name__} under fake tensors "
            f"({type(error).__name__}: {str(error).splitlines()[0][:200]}); its standardization is unchecked",
            stacklevel=3,
        )
        return
    stream = seen["input"]
    if len(stream) != 3 or stream[:2] != ids.shape or seen["first"] != stream or seen["last"] != stream:
        raise _refuse(
            model,
            f"the blocks do not pass a [batch, seq, hidden] stream (block 0 takes {list(stream)} and returns "
            f"{list(seen['first'])}, the last block returns {list(seen['last'])}, for ids {list(ids.shape)})",
        )
    if seen["logits"][:2] != ids.shape:
        raise _refuse(model, f"the logits are {list(seen['logits'])} for ids {list(ids.shape)}")
    for block in model.layers:
        block.returns_tuple = seen["tuple"]

    for envoys in sublayers.values():
        shape: dict[str, Any] = {}
        try:
            with model.scan(ids):
                shape["out"] = first_tensor(envoys[0].output).shape
        except Exception:
            continue  # as above: the scan, not the guess, failed
        if shape["out"] != stream:
            for envoy in envoys:
                envoy._default_reason = (
                    f"its output is {list(shape['out'])}, not the stream's {list(stream)}; a family module points this value at the right place"
                )


# -- sizes: off the modules where the config may not say -----------------------------

def _first(model: "StandardizedTransformer", name: str) -> Standard | None:
    return next((block.__dict__[name] for block in model.layers if isinstance(block.__dict__.get(name), Standard)), None)


def _attention_size(name: str):
    def size(model: "StandardizedTransformer") -> int:
        from ..standardized import StandardizedTransformer

        attention = _first(model, "self_attn")
        if attention is not None:
            try:
                return getattr(attention, name)
            except NotImplementedError:
                pass
        return getattr(StandardizedTransformer, name).fget(model)

    size.__name__ = name
    size.__doc__ = f"The first attention module's ``{name}``, else the root's plain rule over the config."
    return size


num_heads = _attention_size("num_heads")
num_kv_heads = _attention_size("num_kv_heads")
head_dim = _attention_size("head_dim")
qk_head_dim = _attention_size("qk_head_dim")


def intermediate_size(model: "StandardizedTransformer") -> int:
    """The first dense MLP's width off the module (a config can carry a key the model never reads), else the config's ``intermediate_size`` / ``ffn_dim`` / ``ffn_hidden_size`` / ``n_inner``.

    An MLP with ``experts``, a ``router`` or a ``gate`` is a mixture, whose
    module width is one expert's, so the config answers there.
    """
    mlp = _first(model, "mlp")
    if mlp is not None and not any(hasattr(mlp._module, name) for name in ("experts", "router", "gate")):
        try:
            return mlp.intermediate_size
        except NotImplementedError:
            pass
    config = model.config.get_text_config()
    for key in ("intermediate_size", "ffn_dim", "ffn_hidden_size", "n_inner"):
        if isinstance(getattr(config, key, None), int):
            return getattr(config, key)
    raise NotImplementedError(
        f"{type(config).__name__} has no intermediate_size, ffn_dim, ffn_hidden_size or n_inner and no dense MLP module says; "
        "a family module defines `def intermediate_size(model)`"
    )
