"""Llama, end to end: the family the vocabulary is taken from."""

import torch
from suite import FamilySuite, LLAMA_ROWS, rows
from vision_suite import IMAGE, VisionSuite, image_prompt

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
