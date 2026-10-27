"""Qwen3, end to end."""

import pytest

from suite import FamilySuite, LLAMA_ROWS
from vision_suite import WrapperSuite, wrapper_of

from nnterp.families import qwen3


class TestQwen3(FamilySuite):
    REPO = "trl-internal-testing/tiny-Qwen3ForCausalLM"
    FAMILY = qwen3
    NATIVE = LLAMA_ROWS

    def test_head_dim_is_the_configs_not_hidden_over_heads(self, model):
        assert model.head_dim == model.config.head_dim != model.hidden_size // model.num_heads


class TestLightOnOcrWrapper(WrapperSuite):
    """LightOnOCR around Qwen3, built from the tiny text checkpoint's config (no tiny wrapper checkpoint exists): the text stack at
    ``model.language_model``."""

    FAMILY = qwen3

    @pytest.fixture(scope="class")
    def model(self):
        return wrapper_of(TestQwen3.REPO, "LightOnOcrConfig", dict(hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, head_dim=8))
