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

Another engine's families live in a package of their own under this one,
named after the engine: ``nnterp.families.vllm.llama`` is vLLM's implementation
of the ``llama`` model type, looked up with ``lookup("llama", engine="vllm")``.
"""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType

#: ``model_type`` (``"<engine>.<model_type>"`` for another engine's) -> a family passed to `register`,
#: taking precedence over the module of that name.
REGISTRY: dict[str, ModuleType] = {}


class UnsupportedFamily(ValueError):
    """The checkpoint's ``model_type`` has no toolkit: no module of that name here and nothing registered."""


def _key(model_type: str, engine: str | None) -> str:
    """The registry key, which is also the module's name under this package: ``llama``, ``vllm.llama``."""
    return f"{engine}.{model_type}" if engine else model_type


def known(engine: str | None = None) -> list[str]:
    """The shipped families' model types: the modules in this package, or in ``engine``'s package under it."""
    path = __path__ if engine is None else importlib.import_module(f"{__name__}.{engine}").__path__
    return sorted(info.name for info in pkgutil.iter_modules(path) if not info.ispkg)


def lookup(model_type: str, engine: str | None = None) -> ModuleType:
    """The toolkit for ``model_type``: a registered one, else the module of that name, imported on first use.

    ``engine`` names another engine's families (``"vllm"``): its modules are
    their own implementations of the same checkpoints, with their own classes
    and conventions, so they are their own toolkits under the same name.

    Raises:
        UnsupportedFamily: when there is neither.
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
        registered = [key.rpartition(".")[2] for key in REGISTRY if key.rpartition(".")[0] == (engine or "")]
        raise UnsupportedFamily(
            f"no standardization for model_type {model_type!r}{f' on {engine}' if engine else ''}; "
            f"known: {sorted(set(known(engine)) | set(registered))}. "
            f"Add nnterp/families/{key.replace('.', '/')}.py with MODEL_TYPES, RENAME and ENVOYS, or pass a "
            f"module to nnterp.families.register()."
        ) from None


def register(family: ModuleType, engine: str | None = None) -> ModuleType:
    """Add a family toolkit without editing this package.

    ``family`` is any module (or object) with ``MODEL_TYPES``, ``RENAME`` and
    ``ENVOYS`` like the ones here. Its model types go into `REGISTRY`, which
    `lookup` consults before the shipped modules, so a user can also override
    a shipped family. ``engine`` registers it as that engine's
    (``engine="vllm"``). Returns ``family``.
    """
    for model_type in family.MODEL_TYPES:
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
