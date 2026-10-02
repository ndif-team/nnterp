"""The registry: ``config.model_type`` -> the family's standardization toolkit.

A toolkit is one module in this package. Each declares:

* ``MODEL_TYPES``: the ``model_type`` values (from the checkpoint's config)
  the module covers.
* ``RENAME``: an nnsight ``rename`` dict mapping the family's own module names
  onto the standard vocabulary (see `nnter`). Keys are resolved relative to
  every envoy in the tree, so a single-component key such as ``"attn"`` binds
  in every block that has one, and a key that does not resolve anywhere is
  simply skipped.
* ``Layer``, ``Attention`` and ``Mlp``, and ``LinearAttention`` on a hybrid:
  the family's own subclasses of `nnter.components`'s, overriding only what its
  forward spells differently.
* ``ENVOYS``: an nnsight ``envoys`` dict keying those on the family's module
  types (``envoys=`` matches by type or *native* path, never by alias).

A family module is named after the ``model_type`` it covers (``gemma3_text.py``
for ``gemma3_text``), and that is the whole registry: `lookup` imports
``nnter.families.<model_type>`` on first use, so ``import nnter`` loads no
transformers modeling module. To add a family, write the module beside these;
to add one from elsewhere, or to override a shipped one, pass it to `register`.
"""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType

#: ``model_type`` -> a family passed to `register`, taking precedence over the module of that name.
REGISTRY: dict[str, ModuleType] = {}


class UnsupportedFamily(ValueError):
    """The checkpoint's ``model_type`` has no toolkit: no module of that name here and nothing registered."""


def known() -> list[str]:
    """The shipped families' model types: the modules in this package."""
    return sorted(info.name for info in pkgutil.iter_modules(__path__))


def lookup(model_type: str) -> ModuleType:
    """The toolkit for ``model_type``: a registered one, else the module of that name, imported on first use.

    Raises:
        UnsupportedFamily: when there is neither.
    """
    if model_type in REGISTRY:
        return REGISTRY[model_type]
    name = f"{__name__}.{model_type}"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:  # a family module that itself failed to import: a real error
            raise
        raise UnsupportedFamily(
            f"no standardization for model_type {model_type!r}; known: {sorted(set(known()) | set(REGISTRY))}. "
            f"Add nnter/families/{model_type}.py with MODEL_TYPES, RENAME and ENVOYS, or pass a "
            f"module to nnter.families.register()."
        ) from None


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
    """``nnter.families.<model_type>``: the family module, imported on first use."""
    try:
        return importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as error:
        if error.name != f"{__name__}.{name}":
            raise
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(known()))
