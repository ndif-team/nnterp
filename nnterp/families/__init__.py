"""The registry: ``config.model_type`` -> the family's standardization toolkit.

A toolkit is one module in this package. Each declares:

* ``MODEL_TYPES``: the ``model_type`` values (from the checkpoint's config)
  the module covers.
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

A family module is named after the ``model_type`` it covers (``gemma3_text.py``
for ``gemma3_text``), and that is the whole registry: `lookup` imports
``nnterp.families.<model_type>`` on first use, so ``import nnterp`` loads no
transformers modeling module. To add a family, write the module beside these;
to add one from elsewhere, or to override a shipped one, pass it to `register`.
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
    """No toolkit standardizes the checkpoint: it has no family, and the best-effort `default` cannot find what the root needs."""


def known() -> list[str]:
    """The shipped families' model types: the modules in this package, but `default`, which covers none."""
    return sorted(info.name for info in pkgutil.iter_modules(__path__) if info.name != "default")


def lookup(model_type: str) -> ModuleType:
    """The toolkit for ``model_type``: a registered one, else the module of that name, imported on first use.

    With neither, `default` with a warning: its standardization is a guess,
    which it checks at load (raising `UnsupportedFamily` when the guess finds
    no blocks, embedding, final norm or head), and what it could not find or
    trust is unavailable in ``model.support()``.
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
        f"nnterp.families.register()) for a standardization you can rely on.",
        stacklevel=2,
    )
    return importlib.import_module(f"{__name__}.default")


def register(family: ModuleType) -> ModuleType:
    """Add a family toolkit without editing this package.

    ``family`` is any module (or object) with ``MODEL_TYPES``, ``RENAME`` and
    ``ENVOYS`` like the ones here. Its model types go into `REGISTRY`, which
    `lookup` consults before the shipped modules, so a user can also override
    a shipped family. Returns ``family``.
    """
    for model_type in family.MODEL_TYPES:
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
