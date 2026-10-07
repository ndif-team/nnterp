"""The best-effort family, for a ``model_type`` no family module covers.

`nnterp.families.lookup` falls back to this module, with a warning, when the
checkpoint's ``model_type`` has neither a module here nor a registered family;
no ``model_type`` names it.

It serves what a name guess makes safe. ``RENAME`` maps Llama's names over the
spellings the shipped families use for the containers (``model``,
``transformer``, ``gpt_neox``, ``model.decoder``, ``backbone``, a multimodal
``model.language_model`` or ``language_model.model``), the embedding, the
blocks, the final norm, the head, and a block's attention, MLP and norms; a key
that does not resolve on a checkpoint is skipped, so one dict serves every
architecture. ``ENVOYS`` is keyed on the *standard* names: nnsight matches a
string key against a module's alias paths as well as its native one, so
``"self_attn"`` reaches GPT-2's ``attn`` once ``RENAME`` has aliased it, and
``"layers.*"`` reaches every block.

Served: ``layer_output``, the root values, the sizes, and the attention
interior wherever the module calls transformers' shared attention interface
(the base `Attention`'s reads). Not served: ``attention_output``,
``mlp_output`` and `project_on_vocab`, since what a sublayer adds to the
stream, and what follows ``lm_head``, depend on code a name does not show; nor
any mixture, recurrent-mixer or per-family value. Those need a family module
(docs/extending/adding-a-family.md). `check` runs at load and raises
`UnsupportedFamily` when the guess does not hold.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from nnsight.intervention.envoy import Envoy
from nnsight.intervention.source import SourceNotAvailable

from . import UnsupportedFamily
from ..components import INTERFACE, NOT_ON_INTERFACE, Attention, Layer, Mlp, Standard, Unavailable, first_tensor, unavailable

if TYPE_CHECKING:
    from ..standardized import StandardizedTransformer

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

#: Why a sublayer's contribution is not served.
CONTRIBUTION = (
    "the default family cannot tell what this sublayer adds to the stream; "
    "add a family module (docs/extending/adding-a-family.md)"
)
#: Why the logit lens is not served.
LENS = (
    "the default family cannot tell what follows lm_head; "
    "add a family module (docs/extending/adding-a-family.md)"
)


# -- the envoys ----------------------------------------------------------------------

class Layer(Layer):
    """A decoder block as the default finds it.

    ``layer_output`` is the base's: the block's output, or a tuple's first
    element; `check` records which on ``returns_tuple``, so `skip_with` packs
    the stream the way the block does. A block with no module the default
    recognizes as its attention or MLP lists that module's values as
    unavailable, so `support` shows what was not found.
    """

    def support(self) -> dict[str, str | None]:
        rows = super().support()
        for name, envoy_class in (("self_attn", Attention), ("mlp", Mlp)):
            if not isinstance(self.__dict__.get(name), Standard):
                reason = f"no {name} module found under the names the default family knows; a family module names it"
                rows.update((f"{name}.{value}", reason) for value in envoy_class.values())
        return rows


class Attention(Attention):
    """An attention module as the default finds it.

    The interior is the base's, read on transformers' shared attention
    interface; on a forward that makes no such call it is unavailable.
    ``attention_output`` is not served (`CONTRIBUTION`).
    """

    def off_interface(self) -> str | None:
        try:
            names = self.source.names
        except SourceNotAvailable:
            names = ()
        if INTERFACE not in names:
            return NOT_ON_INTERFACE
        return super().off_interface() if hasattr(self._module, "config") else None

    attention_output = unavailable(CONTRIBUTION)


class Mlp(Mlp):
    """A feed-forward module as the default finds it: its sizes only; ``mlp_output`` is not served (`CONTRIBUTION`)."""

    mlp_output = unavailable(CONTRIBUTION)


#: Keyed on the standard names (nnsight matches a string key on alias paths too; ``*`` is any one component).
ENVOYS = {"layers.*": Layer, "self_attn": Attention, "mlp": Mlp}


def project_on_vocab(model: "StandardizedTransformer", hidden: torch.Tensor) -> torch.Tensor:
    """Not served: what the model does past ``lm_head`` (a softcap, a scale) is code a name does not show."""
    raise Unavailable(LENS)


# -- the load-time check -------------------------------------------------------------

#: What the root needs: each is read by a root value or method (`logits` through `lm_head`, `skip_layers`, ...).
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


def _refuse(model: "StandardizedTransformer", what: str, remedy: str) -> UnsupportedFamily:
    return UnsupportedFamily(
        f"the default family cannot standardize {type(model._module).__name__}: {what}. {remedy}, or add a family "
        f"module named after the model_type (docs/extending/adding-a-family.md)."
    )


def _rename(rename: dict[str, str] | None = None) -> str:
    hint = f"; going by the module tree, perhaps rename={rename}" if rename else ""
    return f"Name the native modules with rename={{'<native path>': '<standard name>'}} at load{hint}"


def check(model: "StandardizedTransformer") -> None:
    """Confirm the default's guess on a freshly built model; called by `StandardizedTransformer` at load.

    Raises `UnsupportedFamily` when the root lacks one of `REQUIRED` or no
    block is a `Layer`; when, on a shape-only scan (fake tensors, so nothing
    loads or computes), block 0 does not take and the blocks do not return a
    ``[batch, seq, hidden]`` stream or the logits do not start ``[batch,
    seq]``; and when the scan itself fails, naming the error. Records on each
    block whether it returns a tuple.
    """
    missing = [name for name in REQUIRED if not isinstance(model.__dict__.get(name), Envoy)]
    if not missing and not any(isinstance(block, Layer) for block in model.layers):
        missing = ["layers"]
    if missing:
        raise _refuse(model, f"found no {', '.join(missing)} under any name it knows", _rename(_guess(model, missing)))

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
        raise _refuse(
            model,
            f"its shape check could not run under fake tensors ({type(error).__name__}: {str(error).splitlines()[0][:200]})",
            "Load with arguments under which the forward runs on fake tensors (a mixture of experts: "
            "experts_implementation='batched_mm')",
        ) from error
    stream = seen["input"]
    if len(stream) != 3 or stream[:2] != ids.shape or seen["first"] != stream or seen["last"] != stream:
        raise _refuse(
            model,
            f"the blocks do not pass a [batch, seq, hidden] stream (block 0 takes {list(stream)} and returns "
            f"{list(seen['first'])}, the last block returns {list(seen['last'])}, for ids {list(ids.shape)})",
            _rename(),
        )
    if seen["logits"][:2] != ids.shape:
        raise _refuse(model, f"the logits are {list(seen['logits'])} for ids {list(ids.shape)}", _rename())
    for block in model.layers:
        block.returns_tuple = seen["tuple"]


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
