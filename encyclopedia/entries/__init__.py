"""The hand-written half of each page: one module per family, named after its ``model_type``.

An entry declares::

    MODEL_TYPE   the family's model_type (the module's name)
    TITLE        the page title
    SUBTITLE     one sentence under it
    REFERENCE    the public checkpoint whose config the page is built from (sizes, support())
    PINNED       the tiny checkpoint the tests build the page from
    CHECKPOINTS  public checkpoints of this family, linked from the page
    PALETTE      {"accent", "accent_2", "paper"} from the design's names; any may be left out
    VLLM         whether the family also runs on StandardizedVLLM
    QUIRKS       slugs from build.QUIRKS
    BLOCK        the block schema the visualization draws (see gemma2.py)
    STRIP        notes on the model-level strip, keyed embed / layers / norm / head / logits
    NOTES        markdown: the open-form notes for interpretability

``load_all`` imports every module here but this one.
"""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType


def names() -> list[str]:
    return sorted(info.name for info in pkgutil.iter_modules(__path__))


def load(model_type: str) -> ModuleType:
    entry = importlib.import_module(f"{__name__}.{model_type}")
    assert entry.MODEL_TYPE == model_type, (entry.MODEL_TYPE, model_type)
    return entry


def load_all() -> list[ModuleType]:
    return [load(name) for name in names()]
