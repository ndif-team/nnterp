"""Ministral 3, end to end."""

import torch
from suite import FamilySuite, LLAMA_ROWS, rows
from vision_suite import PixtralSuite

from nnterp.families import ministral3


class TestMinistral3(FamilySuite):
    REPO = "yujiepan/ministral-3-tiny-random"
    FAMILY = ministral3
    NATIVE = LLAMA_ROWS


class TestMistral3Wrapper(TestMinistral3):
    """Mistral 3 loaded as the wrapper with its processor: the text stack at ``model.language_model``."""

    REPO = "yujiepan/mistral-3-tiny-random"
    NATIVE = rows("model.language_model", "layers", "embed_tokens", "norm")
    LOAD_KWARGS = {"task": "image-text-to-text", "dtype": torch.float32}   # in bf16 a zeroed query head moves no logit


class TestMistral3Vision(PixtralSuite):
    """Mistral 3's Pixtral tower and patch-merging projector, and the tower's image values."""

    REPO = "yujiepan/mistral-3-tiny-random"
    FAMILY = ministral3
    TEXT_REPO = TestMinistral3.REPO
