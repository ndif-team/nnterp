"""What holds on every host of a vision tower: one module per tower that two or more families host.

A tower module declares::

    TITLE                the tower's name, shown on the page and the index ("CLIP")
    VISION_CONFIG_TYPES  the ``config.vision_config.model_type`` values it covers
    MODULE_CLASSES       the class names ``model.vision`` may have; the build asserts one of them
    BLOCK                the tower block the diagram draws, in an entry's ``BLOCK`` format; ``detail``
                         is formatted with the tower's sizes and its ``vision_config`` keys
    ROWS                 what a row of ``Patches`` is, and the patch axis
    MASKING              what a patch attends to
    POSITIONS            how positions enter the tower's stream
    NORM                 the tower's final norm, and whether it is ``vision.norm``
    QUIRKS               slugs from ``build.QUIRKS``
    NOTES                markdown: facts of the tower on every host, nothing host-specific

ROWS, MASKING, POSITIONS and NORM are one or two sentences each, code names in backticks.

A tower one family hosts stays inline, as ``WRAPPERS[<wrapper>]["tower"]`` in the entry, a dict with the
same fields (``VISION_CONFIG_TYPES`` may be left out there). `resolve` returns either form as one dict.
"""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType
from typing import Any

FIELDS = ("TITLE", "VISION_CONFIG_TYPES", "MODULE_CLASSES", "BLOCK", "ROWS", "MASKING", "POSITIONS", "NORM", "QUIRKS", "NOTES")


def modules() -> list[ModuleType]:
    return [importlib.import_module(f"{__name__}.{info.name}") for info in sorted(pkgutil.iter_modules(__path__), key=lambda i: i.name)]


def normalize(slug: str, fields: dict[str, Any]) -> dict[str, Any]:
    """A tower as one dict: the fields lower-cased, and ``slug`` (the module's name, or ``<family>.<wrapper>`` inline)."""
    missing = [name for name in FIELDS if name not in fields and name != "VISION_CONFIG_TYPES"]
    assert not missing, f"tower {slug!r} lacks {missing}"
    return {"slug": slug, **{name.lower(): fields.get(name, []) for name in FIELDS}}


def resolve(entry: ModuleType, wrapper: str, vision_type: str) -> dict[str, Any]:
    """The tower of an entry's ``wrapper``: inline in ``WRAPPERS[wrapper]["tower"]``, else the module covering ``vision_type``."""
    inline = entry.WRAPPERS[wrapper].get("tower")
    if inline is not None:
        return normalize(f"{entry.MODEL_TYPE}.{wrapper}", inline)
    found = [module for module in modules() if vision_type in module.VISION_CONFIG_TYPES]
    assert len(found) == 1, f"{entry.MODEL_TYPE}: {len(found)} tower modules cover vision_config.model_type {vision_type!r}"
    module = found[0]
    return normalize(module.__name__.rsplit(".", 1)[1], {name: getattr(module, name) for name in FIELDS})
