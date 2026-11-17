"""The registry: ``config.model_type`` -> the family's standardization toolkit.

A toolkit is one module in this package, named after the ``model_type``
(from the checkpoint's config) it covers. Each declares:

* ``RENAME``: an nnsight ``rename`` dict mapping the family's own module names
  onto the standard vocabulary (see `nnterp`).
* ``Layer``, ``Attention`` and ``Mlp``: the family's own subclasses of
  `nnterp.components`'s, overriding only what its forward spells differently.
* ``ENVOYS``: an nnsight ``envoys`` dict keying those on the family's module
  types.

`lookup` imports ``nnterp.families.<model_type>`` on first use, so
``import nnterp`` loads no transformers modeling module. To add a family,
write the module beside these; to add one from elsewhere, or to override a
shipped one, pass it to `register`; to use one for a single load, pass it as
``StandardizedTransformer(..., family=)``. A ``model_type`` with neither gets
`default`, the best-effort family, with a warning.

Another engine's families live in a package of their own under this one,
named after the engine: ``nnterp.families.vllm.llama``, looked up with
``lookup("llama", engine="vllm")``. There is no default on another engine.
"""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
import warnings
from types import ModuleType
from typing import Any

#: ``model_type`` (``"<engine>.<model_type>"`` for another engine's) -> a family passed to `register`,
#: taking precedence over the module of that name.
REGISTRY: dict[str, ModuleType] = {}


class UnsupportedFamily(ValueError):
    """No toolkit standardizes the checkpoint: it has no family, and the best-effort `default` cannot standardize it."""


def _key(model_type: str, engine: str | None) -> str:
    """The registry key, which is also the module's name under this package: ``llama``, ``vllm.llama``."""
    return f"{engine}.{model_type}" if engine else model_type


def covers(model_type: str, engine: str | None = None) -> bool:
    """Whether a family covers ``model_type`` on ``engine``: one registered for it, or a module of that name here."""
    key = _key(model_type, engine)
    if key in REGISTRY:
        return True
    try:  # a dotted model_type (kosmos-2.5) reads as a package path; no module here has one
        return importlib.util.find_spec(f"{__name__}.{key}") is not None
    except ModuleNotFoundError:
        return False


def resolve(config: Any, engine: str | None = None) -> ModuleType:
    """The family for a config: its own ``model_type`` where a family covers it, else its text model's.

    A multimodal wrapper usually resolves through its ``text_config`` (Llava to
    ``llama``). A wrapper with a family of its own, named after the wrapper,
    resolves to that family whatever its text config says (``kimi_k25``).
    """
    text = getattr(config, "text_config", config)
    if text is not config and covers(config.model_type, engine):
        return lookup(config.model_type, engine)
    return lookup(text.model_type, engine)


def known(engine: str | None = None) -> list[str]:
    """The shipped families' model types: the modules in this package (but `default`, which covers none), or in ``engine``'s package under it."""
    path = __path__ if engine is None else importlib.import_module(f"{__name__}.{engine}").__path__
    return sorted(info.name for info in pkgutil.iter_modules(path) if not info.ispkg and info.name != "default")


def lookup(model_type: str, engine: str | None = None) -> ModuleType:
    """The toolkit for ``model_type``: a registered one, else the module of that name, imported on first use.

    With neither, on transformers, `default` with a warning: a name guess,
    which it checks at load. ``engine`` names another engine's families
    (``"vllm"``), which have no default.

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
