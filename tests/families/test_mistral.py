"""Mistral, end to end."""

import pytest

from suite import FamilySuite, LLAMA_ROWS
from vision_suite import WrapperSuite, wrapper_of

from nnterp.families import mistral


class TestMistral(FamilySuite):
    REPO = "hf-internal-testing/tiny-random-MistralForCausalLM"
    FAMILY = mistral
    NATIVE = LLAMA_ROWS


class TestMistral3Wrapper(WrapperSuite):
    """Mistral 3 (Mistral Small 3.1 / 3.2) around Mistral, built from the tiny text checkpoint's config (no tiny wrapper checkpoint exists): the text stack at
    ``model.language_model``."""

    FAMILY = mistral

    @pytest.fixture(scope="class")
    def model(self):
        return wrapper_of(TestMistral.REPO, "Mistral3Config", dict(hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, head_dim=8))
