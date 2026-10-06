"""Llama, end to end: the family the vocabulary is taken from."""

import pytest
import torch
from suite import FamilySuite, LLAMA_ROWS, rows
from vision_suite import IMAGE, VisionSuite, align_processor, clip_rows, image_prompt

from nnterp import StandardizedTransformer, Unavailable
from nnterp.families import llama


class TestLlama(FamilySuite):
    REPO = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    FAMILY = llama
    NATIVE = LLAMA_ROWS


class TestLlavaWrapper(TestLlama):
    """Llava 1.5 loaded as the wrapper with its processor: the text stack at ``model.language_model``, the whole suite on it."""

    REPO = "trl-internal-testing/tiny-LlavaForConditionalGeneration"
    NATIVE = rows("model.language_model", "layers", "embed_tokens", "norm")
    LOAD_KWARGS = {"task": "image-text-to-text"}


class TestIdefics3Wrapper(TestLlama):
    """Idefics 3 (the SmolVLM layout): the text stack at ``model.text_model``; no vision names yet."""

    REPO = "trl-internal-testing/tiny-Idefics3ForConditionalGeneration"
    NATIVE = rows("model.text_model", "layers", "embed_tokens", "norm")
    LOAD_KWARGS = {"task": "image-text-to-text"}


class TestLlavaVision(VisionSuite):
    """Llava 1.5's CLIP tower and projector, and the tower's image values."""

    REPO = "trl-internal-testing/tiny-LlavaForConditionalGeneration"
    FAMILY = llama
    TEXT_REPO = TestLlama.REPO
    VISION_NATIVE = {
        "vision": "model.vision_tower",
        "vision.layers": "model.vision_tower.encoder.layers",
        "vision.patch_embed": "model.vision_tower.embeddings.patch_embedding",
        "vision.layers.0.self_attn": "model.vision_tower.encoder.layers.0.self_attn",
        "vision.layers.0.mlp": "model.vision_tower.encoder.layers.0.mlp",
        "vision.layers.0.input_layernorm": "model.vision_tower.encoder.layers.0.layer_norm1",
        "vision.layers.0.post_attention_layernorm": "model.vision_tower.encoder.layers.0.layer_norm2",
        "projector": "model.multi_modal_projector",
    }

    def test_clip_has_no_final_norm_over_the_patches(self, model):
        """CLIP's ``post_layernorm`` norms the pooled CLS token only, so it is not ``vision.norm``."""
        assert "norm" not in model.vision._aliases
        assert model.vision.post_layernorm is model.get("model.vision_tower.post_layernorm")

    def test_the_projector_takes_the_second_to_last_block_without_cls(self, model):
        with model.trace(image_prompt(model), images=[IMAGE]):
            stream = model.vision.layers[-2].layer_output.save()
            fed = model.projector.input.save()
        assert model.config.vision_feature_layer == -2
        assert torch.equal(fed, stream[:, 1:])


def tiny_v2_processor(model):
    """The hf-tiny-v2 Llava-class processors: ``crop_size`` has edge keys where center-cropping needs ``height`` and
    ``width``, and ``patch_size`` and ``vision_feature_select_strategy`` are unset, so the image token count fails."""
    from transformers.image_utils import SizeDict

    side = model.processor.image_processor.crop_size.shortest_edge
    align_processor(
        model, patch_size=model.config.vision_config.patch_size, num_additional_image_tokens=1,  # CLIP's CLS token
        vision_feature_select_strategy=getattr(model.config, "vision_feature_select_strategy", "default"),  # VipLlava drops CLS
        image_processor={"crop_size": SizeDict(height=side, width=side)},
    )


class TestLlavaNextVision(VisionSuite):
    """LLaVA-NeXT around Llama: CLIP over the image's crops, and a projector output the wrapper unpads and adds newline tokens to."""

    REPO = "hf-tiny-v2/tiny-random-LlavaNextForConditionalGeneration"
    FAMILY = llama
    TEXT_REPO = TestLlama.REPO
    VISION_NATIVE = clip_rows()

    @staticmethod
    def fix_processor(model):
        """As `tiny_v2_processor`, and the processor's crop grid is the 336-pixel default where the model's is ``[[8, 8]]``."""
        tiny_v2_processor(model)
        model.processor.image_processor.image_grid_pinpoints = model.config.image_grid_pinpoints

    def test_image_features_are_not_the_projector_output(self, model):
        """What is scattered is the unpadded projector output with a newline token per row, so the projector read would be wrong."""
        with model.trace(image_prompt(model), images=[IMAGE]):
            projected = model.projector.output.save()
            features = model.vision.image_features.save()
        assert projected.shape[0] > 1  # the base image and its crops
        assert features.shape[0] != projected.shape[0] * projected.shape[1]
        newline = model.get("model").image_newline.detach()
        assert (features == newline).all(-1).any()


def test_image_features_need_an_image_scatter():
    """Without an `ImageScatter` on the wrapper's model, where the features enter the text stream is unknown: the value says so."""
    from nnsight.intervention.envoy import Envoy
    from transformers.models.llava.modeling_llava import LlavaModel

    model = StandardizedTransformer(TestLlavaVision.REPO, task="image-text-to-text", envoys={LlavaModel: Envoy})
    reason = model.vision.support()["image_features"]
    assert "keys no ImageScatter on the 'llava' wrapper" in reason
    assert model.vision.support()["image_token_mask"] is None
    with pytest.raises(Unavailable, match="ImageScatter"):
        model.vision.image_features
