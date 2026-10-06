"""Mistral, end to end."""

import pytest

from suite import FamilySuite, LLAMA_ROWS
from vision_suite import IMAGE, VisionSuite, WrapperSuite, clip_rows, image_prompt, wrapper_of

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


class TestLlavaNextMistralVision(VisionSuite):
    """LLaVA-NeXT around Mistral (``llava-v1.6-mistral``): CLIP over the image's crops; ``image_features`` is the unpadded
    projector output with a newline token per row, as the text model receives it."""

    REPO = "trl-internal-testing/tiny-LlavaNextForConditionalGeneration"
    FAMILY = mistral
    TEXT_REPO = TestMistral.REPO
    VISION_NATIVE = clip_rows()

    def test_image_features_are_not_the_projector_output(self, model):
        with model.trace(image_prompt(model), images=[IMAGE]):
            projected = model.projector.output.save()
            features = model.vision.image_features.save()
        assert projected.shape[0] > 1  # the base image and its crops
        assert features.shape[0] != projected.shape[0] * projected.shape[1]
        newline = model.get("model").image_newline.detach()
        assert (features == newline).all(-1).any()
