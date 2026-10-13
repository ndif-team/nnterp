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
``StandardizedTransformer(..., family=)`` or ``StandardizedVLLM(..., family=)``.
A ``model_type`` with neither gets `default`, the best-effort family, with a
warning; it is not one of `known`.

Another engine's families live in a package of their own under this one,
named after the engine: ``nnterp.families.vllm.llama`` is vLLM's implementation
of the ``llama`` model type, looked up with ``lookup("llama", engine="vllm")``.
There is no default on another engine: a type it has no family for raises.
"""

from __future__ import annotations

import importlib
import pkgutil
import warnings
from types import ModuleType

#: ``model_type`` (``"<engine>.<model_type>"`` for another engine's) -> a family passed to `register`,
#: taking precedence over the module of that name.
REGISTRY: dict[str, ModuleType] = {}


class UnsupportedFamily(ValueError):
    """No toolkit standardizes the checkpoint: it has no family, and the best-effort `default` cannot standardize it."""


def _key(model_type: str, engine: str | None) -> str:
    """The registry key, which is also the module's name under this package: ``llama``, ``vllm.llama``."""
    return f"{engine}.{model_type}" if engine else model_type


def known(engine: str | None = None) -> list[str]:
    """The shipped families' model types: the modules in this package (but `default`, which covers none), or in ``engine``'s package under it."""
    path = __path__ if engine is None else importlib.import_module(f"{__name__}.{engine}").__path__
    return sorted(info.name for info in pkgutil.iter_modules(path) if not info.ispkg and info.name != "default")


def lookup(model_type: str, engine: str | None = None) -> ModuleType:
    """The toolkit for ``model_type``: a registered one, else the module of that name, imported on first use.

    With neither, on transformers, `default` with a warning: its
    standardization is a name guess, which it checks at load (raising
    `UnsupportedFamily` when the guess finds no blocks, embedding, final norm
    or head, or the stream's shape does not hold), and what a guess cannot
    make safe (the contributions, the logit lens) is unavailable.

    ``engine`` names another engine's families (``"vllm"``): its modules are
    their own implementations of the same checkpoints, with their own classes
    and conventions, so they are their own toolkits under the same name.
    `default` is written against transformers' modules, so it is no fallback
    there: an engine's lookup with no family raises.

    Raises:
        UnsupportedFamily: on another engine, when there is neither.
    """
    key = _key(model_type, engine)
    if key in REGISTRY:
        return REGISTRY[key]
    name = f"{__name__}.{key}"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:  # a family module that itself failed to import: a real error
            raise
    if engine is not None:
        registered = [key.rpartition(".")[2] for key in REGISTRY if key.rpartition(".")[0] == engine]
        raise UnsupportedFamily(
            f"no standardization for model_type {model_type!r} on {engine}; "
            f"known: {sorted(set(known(engine)) | set(registered))}. "
            f"Add nnterp/families/{key.replace('.', '/')}.py with RENAME and ENVOYS, or pass a family to "
            f"nnterp.families.register(family, {model_type!r}, engine={engine!r})."
        )
    warnings.warn(
        f"nnterp has no family for model_type {model_type!r}; the default family standardizes it as a best-effort "
        f"guess. Check model.support() for what it found, and add nnterp/families/{model_type}.py (or "
        f"nnterp.families.register(family, {model_type!r})) for a standardization you can rely on.",
        stacklevel=2,
    )
    return importlib.import_module(f"{__name__}.default")


def register(family: ModuleType, *model_types: str, engine: str | None = None) -> ModuleType:
    """Add a family toolkit without editing this package.

    ``family`` is any module (or object) with ``RENAME`` and ``ENVOYS`` like
    the ones here. It covers ``model_types``; with none given, the type its
    ``__name__`` names, as a shipped module's file name does
    (``register(my_pkg.zamba)`` covers ``zamba``). The types go into
    `REGISTRY`, which `lookup` consults before the shipped modules, so a user
    can also override a shipped family. ``engine`` registers it as that
    engine's (``engine="vllm"``). Returns ``family``.

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
        REGISTRY[_key(model_type, engine)] = family
    return family


def all_families(engine: str | None = None) -> list[ModuleType]:
    """Every shipped family, imported: for tooling and tests, not for a load."""
    return [lookup(name, engine) for name in known(engine)]


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
