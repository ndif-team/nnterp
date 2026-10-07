"""Gemma 1, end to end."""

import pytest
from suite import FamilySuite, LLAMA_ROWS, PROMPT
from vision_suite import VisionSuite, WrapperSuite, align_processor, siglip_rows

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


class TestPaliGemmaVision(VisionSuite):
    """PaliGemma's SigLIP tower and linear projector, and the tower's image values.

    ``hf-tiny-v2``: the trl tiny wrapper (the text test's) projects to 2048 where its text model is 16 wide, so its
    image path fails in transformers' own forward.
    """

    REPO = "hf-tiny-v2/tiny-random-PaliGemmaForConditionalGeneration"
    FAMILY = gemma
    TEXT_REPO = TestGemma.REPO
    VISION_NATIVE = siglip_rows()

    @staticmethod
    def fix_processor(model):
        vision = model.config.vision_config
        align_processor(model, image_seq_length=(vision.image_size // vision.patch_size) ** 2)

    def text_input(self, model):
        return dict(model.tokenizer(PROMPT, return_tensors="pt"))
