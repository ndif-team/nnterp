"""Mistral, end to end."""

import pytest
import torch

from suite import FamilySuite, LLAMA_ROWS
from vision_suite import IMAGE, PixtralSuite, VisionSuite, WrapperSuite, clip_rows, image_prompt, wrapper_of

from nnterp import StandardizedTransformer
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


class TestMistral3Vision(PixtralSuite):
    """Mistral 3 around Mistral (Mistral Small 3.1 / 3.2): the Pixtral tower and patch-merging projector."""

    REPO = "hf-tiny-v2/tiny-random-Mistral3ForConditionalGeneration"
    FAMILY = mistral
    TEXT_REPO = TestMistral.REPO

    @staticmethod
    def fix_processor(model):
        """The tiny checkpoint's processor keeps Pixtral-12B's 16-pixel patches unmerged where the tower's are 6 pixels merged
        2x2, and the config's ``image_token_id`` (1) is not the processor's ``[IMG]``."""
        processor, config = model.processor, model.config
        processor.patch_size = processor.image_processor.patch_size = config.vision_config.patch_size
        processor.spatial_merge_size = config.spatial_merge_size
        config.image_token_id = processor.image_token_id


class TestLlavaPixtralVision(PixtralSuite):
    """Pixtral-12B's Llava wrapper (``llava``, Mistral text, Pixtral tower), built small from ``mistral-community/pixtral-12b``'s
    config with random weights, with that checkpoint's processor (no tiny checkpoint exists)."""

    REPO = "mistral-community/pixtral-12b"
    FAMILY = mistral
    TEXT_REPO = TestMistral.REPO

    def small(self):
        """The wrapper from the checkpoint's config, shrunk, with random weights."""
        from transformers import AutoConfig, AutoModelForImageTextToText

        config = AutoConfig.from_pretrained(self.REPO)
        config.text_config.update(dict(hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8))
        config.vision_config.update(dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, head_dim=16))
        torch.manual_seed(0)
        return AutoModelForImageTextToText.from_config(config, attn_implementation="eager").eval()

    @pytest.fixture(scope="class")
    def model(self):
        from transformers import AutoProcessor

        return StandardizedTransformer(self.small(), processor=AutoProcessor.from_pretrained(self.REPO), task="image-text-to-text")

    def text_generation_load(self):
        from transformers import AutoTokenizer

        return StandardizedTransformer(self.small(), tokenizer=AutoTokenizer.from_pretrained(self.REPO))

    def test_the_wrapper_is_llava(self, model):
        assert model.config.model_type == "llava" and model.config.vision_config.model_type == "pixtral"


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
