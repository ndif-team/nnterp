"""Gemma 1, end to end."""

import pytest
from suite import FamilySuite, LLAMA_ROWS, PROMPT
from vision_suite import WrapperSuite

from nnterp import StandardizedTransformer

from nnterp.families import gemma


class TestGemma(FamilySuite):
    REPO = "trl-internal-testing/tiny-GemmaForCausalLM"
    FAMILY = gemma
    NATIVE = LLAMA_ROWS


class TestPaliGemmaWrapper(WrapperSuite):
    """PaliGemma: the text stack at ``model.language_model``. Its processor refuses a prompt without an image, so the
    text-only trace takes the tokenizer's encoding."""

    FAMILY = gemma

    @pytest.fixture(scope="class")
    def model(self):
        return StandardizedTransformer(
            "trl-internal-testing/tiny-PaliGemmaForConditionalGeneration", task="image-text-to-text", dispatch=True, attn_implementation="eager",
        )

    def text_input(self, model):
        return dict(model.tokenizer(PROMPT, return_tensors="pt"))
