"""The registry: ``config.model_type`` -> the family's standardization toolkit.

A toolkit is one module in this package, named after the ``model_type``
(from the checkpoint's config) it covers. Each declares:

* ``RENAME``: an nnsight ``rename`` dict mapping the family's own module names
  onto the standard vocabulary (see `nnterp`). Keys are resolved relative to
  every envoy in the tree, so a single-component key such as ``"attn"`` binds
  in every block that has one, and a key that does not resolve anywhere is
  simply skipped.
* ``Layer``, ``Attention`` and ``Mlp``, and ``LinearAttention`` on a hybrid:
  the family's own subclasses of `nnterp.components`'s, overriding only what its
  forward spells differently.
* ``ENVOYS``: an nnsight ``envoys`` dict keying those on the family's module
  types (``envoys=`` matches by type or *native* path, never by alias).

The module's name is its ``model_type`` (``gemma3_text.py`` covers
``gemma3_text``), and that is the whole registry: `lookup` imports
``nnterp.families.<model_type>`` on first use, so ``import nnterp`` loads no
transformers modeling module. To add a family, write the module beside these;
to add one from elsewhere, or to override a shipped one, pass it to `register`
with the model types it covers; to use one for a single load, pass it as
``StandardizedTransformer(..., family=)``.
A ``model_type`` with neither gets `default`, the best-effort family, with a
warning; it is not one of `known`.
"""

from __future__ import annotations

import importlib
import pkgutil
import warnings
from types import ModuleType

#: ``model_type`` -> a family passed to `register`, taking precedence over the module of that name.
REGISTRY: dict[str, ModuleType] = {}


class UnsupportedFamily(ValueError):
    """No toolkit standardizes the checkpoint: it has no family, and the best-effort `default` cannot standardize it."""


def known() -> list[str]:
    """The shipped families' model types: the modules in this package, but `default`, which covers none."""
    return sorted(info.name for info in pkgutil.iter_modules(__path__) if info.name != "default")


def lookup(model_type: str) -> ModuleType:
    """The toolkit for ``model_type``: a registered one, else the module of that name, imported on first use.

    With neither, `default` with a warning: its standardization is a name
    guess, which it checks at load (raising `UnsupportedFamily` when the guess
    finds no blocks, embedding, final norm or head, or the stream's shape does
    not hold), and what a guess cannot make safe (the contributions, the logit
    lens) is unavailable.
    """
    if model_type in REGISTRY:
        return REGISTRY[model_type]
    name = f"{__name__}.{model_type}"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:  # a family module that itself failed to import: a real error
            raise
    warnings.warn(
        f"nnterp has no family for model_type {model_type!r}; the default family standardizes it as a best-effort "
        f"guess. Check model.support() for what it found, and add nnterp/families/{model_type}.py (or "
        f"nnterp.families.register(family, {model_type!r})) for a standardization you can rely on.",
        stacklevel=2,
    )
    return importlib.import_module(f"{__name__}.default")


def register(family: ModuleType, *model_types: str) -> ModuleType:
    """Add a family toolkit without editing this package.

    ``family`` is any module (or object) with ``RENAME`` and ``ENVOYS`` like
    the ones here. It covers ``model_types``; with none given, the type its
    ``__name__`` names, as a shipped module's file name does
    (``register(my_pkg.zamba)`` covers ``zamba``). The types go into
    `REGISTRY`, which `lookup` consults before the shipped modules, so a user
    can also override a shipped family. Returns ``family``.

    Raises:
        TypeError: when no type is given and ``family`` has no ``__name__``
            (a ``types.SimpleNamespace``): pass them, ``register(ns, "zamba")``.
    """
    if not model_types:
        name = getattr(family, "__name__", None)
        if not name:
            raise TypeError(
                f"register() needs the model types the family covers: a {type(family).__name__} has no __name__ "
                f"to take one from. Pass them: register(family, 'my_model_type')."
            )
        model_types = (name.rsplit(".", 1)[-1],)
    for model_type in model_types:
        REGISTRY[model_type] = family
    return family


def all_families() -> list[ModuleType]:
    """Every shipped family, imported: for tooling and tests, not for a load."""
    return [lookup(name) for name in known()]


def __getattr__(name: str) -> ModuleType:
    """``nnterp.families.<model_type>``: the family module, imported on first use."""
    try:
        return importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as error:
        if error.name != f"{__name__}.{name}":
            raise
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(known()))
