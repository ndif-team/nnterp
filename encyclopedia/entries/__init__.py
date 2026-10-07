"""The hand-written half of each page: one module per family, named after its ``model_type``.

An entry declares::

    MODEL_TYPE   the family's model_type (the module's name)
    TITLE        the page title
    SUBTITLE     one sentence under it
    REFERENCE    the public checkpoint the page opens on
    ORG          optional: the Hub org the index files the family under (the reference's author by default)
    PINNED       the tiny checkpoint the tests build the page from
    CHECKPOINTS  public checkpoints of this family, the choices of the page's selector, each built from its config
    PALETTE      {"hue": degrees} picks the base hue the five colours are generated from, or
                 {"colors": [five hex fills], "deeps": [five hex, optional]} gives them outright;
                 "paper" tints the page; the hue is hues.py's (run it after adding a family)
    VLLM         whether the family also runs on StandardizedVLLM
    QUIRKS       slugs from build.QUIRKS
    BLOCK        the block schema the visualization draws (see gemma2.py; kimi_linear.py for blocks that differ;
                 falcon.py for a list of (predicate on the config, block) pairs, one layout per checkpoint)
    load         optional: load(checkpoint, **kwargs) -> StandardizedTransformer, for a checkpoint a
                 repo id alone does not build on the meta device
    GREYED       optional: {repo id: reason}, checkpoints listed greyed out with that reason and not built
    STRIP        notes on the model-level strip, keyed embed / layers / norm / head / logits
    NOTES        markdown: the open-form notes for interpretability
    WRAPPERS     optional: the family's vision-language wrappers by config.model_type, each {title, pinned,
                 projector, quirks?, notes, tower?}; the tower's own facts are in encyclopedia/vision/

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
