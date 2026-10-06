"""Gemma 4 unified, end to end: Gemma-4's sandwich and layer_scalar, KV sharing, no per-layer embeddings or experts;
and the wrapper's encoder-free image embedder under ``vision``."""

import glob
import os
import tempfile

import pytest
import torch
from test_gemma4_text import Gemma4Suite
from vision_suite import IMAGE, TOWER_VALUES, VisionSuite, image_prompt

from nnterp import StandardizedTransformer, Unavailable
from nnterp.components import Vision
from nnterp.families import gemma4_unified_text

TEXT_REPO = "hf-tiny-v2/tiny-random-Gemma4UnifiedForCausalLM"


class TestGemma4Unified(Gemma4Suite):
    REPO = TEXT_REPO
    FAMILY = gemma4_unified_text

    def test_no_per_layer_output(self, model):
        assert "per_layer_output" not in model.support()
        assert not hasattr(gemma4_unified_text.Layer, "per_layer_output")


def _unified_config():
    """``google/gemma-4-12B``'s config (the one ``gemma4_unified`` release) with the tiny text model and a 16-wide embedder."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained("google/gemma-4-12B")
    config.text_config = AutoConfig.from_pretrained(TEXT_REPO)
    config.vision_config.mm_embed_dim = config.vision_config.output_proj_dims = 16
    return config


def _unified_checkpoint():
    """A tiny ``Gemma4UnifiedForConditionalGeneration`` with its processor, random weights; written once per text snapshot.

    No tiny wrapper is published. The config is `_unified_config`; the
    processor is the tiny text checkpoint's tokenizer, chat template and
    audio feature extractor with the default image and video processors
    (12B's own: 48-pixel merged patches, at most 280 per image).
    """
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from transformers.models.gemma4_unified import (
        Gemma4UnifiedAudioFeatureExtractor, Gemma4UnifiedImageProcessor, Gemma4UnifiedProcessor,
        Gemma4UnifiedVideoProcessor,
    )

    snapshot = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{TEXT_REPO.replace('/', '--')}/snapshots/*"))[0]
    built = os.path.join(tempfile.gettempdir(), f"nnterp-gemma4-unified-{os.path.basename(snapshot)}")
    if os.path.exists(os.path.join(built, "processor_config.json")):
        return built
    torch.manual_seed(0)
    partial = tempfile.mkdtemp(prefix="nnterp-gemma4-unified-")
    AutoModelForImageTextToText.from_config(_unified_config()).save_pretrained(partial)
    Gemma4UnifiedProcessor(
        feature_extractor=Gemma4UnifiedAudioFeatureExtractor.from_pretrained(snapshot),
        image_processor=Gemma4UnifiedImageProcessor(),
        tokenizer=AutoTokenizer.from_pretrained(snapshot),
        video_processor=Gemma4UnifiedVideoProcessor(),
        chat_template=open(os.path.join(snapshot, "chat_template.jinja")).read(),
    ).save_pretrained(partial)
    try:
        os.replace(partial, built)
    except OSError:  # another process built it first
        pass
    return built


class TestGemma4UnifiedVision(VisionSuite):
    """The encoder-free embedder: ``vision`` with no blocks, ``projector`` its projection, ``image_features`` read at the scatter."""

    REPO = _unified_checkpoint()
    FAMILY = gemma4_unified_text
    TEXT_REPO = TEXT_REPO
    TEXT_GENERATION_BUILDS_WRAPPER = True
    VISION_NATIVE = {
        "vision": "model.embed_vision",
        "vision.patch_embed": "model.embed_vision.patch_dense",
        "projector": "model.embed_vision.multimodal_embedder",
    }

    @pytest.fixture(scope="class")
    def positions(self, model):
        """The processor's patch positions for the image: ``(-1, -1)`` on the padded rows."""
        return model.processor(text=image_prompt(model), images=[IMAGE], return_tensors="pt")["image_position_ids"]

    # -- what an encoder-free embedder has instead of a tower ---------------------------

    def test_tower_envoy_classes(self, model):
        assert type(model.vision) is gemma4_unified_text.Vision and isinstance(model.vision, Vision)
        assert "layers" not in model.vision._aliases and not hasattr(model.vision._module, "layers")
        assert model.projector is model.vision.multimodal_embedder

    def test_sizes_are_the_towers(self, model):
        vision, config = model.vision, model.config.vision_config
        assert (vision.num_layers, vision.hidden_size, vision.patch_size) == (0, config.mm_embed_dim, config.model_patch_size)
        for name in ("num_heads", "head_dim", "intermediate_size"):
            with pytest.raises(Unavailable, match="encoder-free"):
                getattr(vision, name)
        with pytest.raises(Unavailable, match="any resolution"):
            vision.image_size
        text = model.config.get_text_config()
        assert model.num_layers == len(model.layers) == text.num_hidden_layers and model.hidden_size == text.hidden_size

    def test_support_lists_the_tower_values(self, model):
        vision = model.vision.support()
        assert set(vision) == set(TOWER_VALUES) and all(reason is None for reason in vision.values()), vision
        support = model.support()
        assert {name.removeprefix("vision."): reason for name, reason in support.items() if name.startswith("vision.")} == vision
        assert not {"image_token_mask", "image_features"} & set(support)
        with pytest.raises(IndexError, match="no blocks"):
            model.vision.support(layer=0)

    def test_layouts(self, model):
        from nnterp.components import ImageFeatures, ImageTokenMask, Patches

        cls = type(model.vision)
        assert cls.image_token_mask.layout is ImageTokenMask and cls.image_features.layout is ImageFeatures
        assert cls.patch_embeddings.layout is Patches and cls.tower_output.layout is Patches

    def test_a_tower_write_moves_the_image_features(self, model, clean):
        with model.trace(image_prompt(model), images=[IMAGE]):
            model.vision.tower_output = torch.randn_like(model.vision.tower_output)
            features = model.vision.image_features.save()
        assert not torch.allclose(features, clean["features"])

    def test_no_tower_name_binds_on_the_text_model(self, model):
        assert all(type(layer) is self.FAMILY.Layer for layer in model.layers)
        tower = {"vision", "projector", "patch_embed"}
        assert not tower & set(model.layers[0]._aliases) and not tower & set(model.layers._aliases)

    # No blocks: the block checks have nothing to run on.
    test_tower_contribution_identity = None
    test_tower_pattern_sums_to_one_over_keys = None

    # -- the padding the wrapper strips --------------------------------------------------

    def test_patch_embeddings_and_tower_output(self, model, positions):
        vision = model.vision
        with model.trace(image_prompt(model), images=[IMAGE]):
            patches = vision.patch_embeddings.save()
            dense = vision.patch_embed.output.save()
            out = vision.tower_output.save()
            fed = model.projector.input.save()
        assert patches.shape == out.shape == (1, positions.shape[1], vision.hidden_size)
        assert torch.equal(patches, dense) and torch.equal(out, fed)

    def test_image_features_are_the_projection_without_the_padding(self, model, positions, clean):
        """The projector runs on every padded row; the wrapper strips them: ``image_features == projector.output[valid]``."""
        valid = (positions != -1).all(-1)
        assert valid.any() and not valid.all()
        with model.trace(image_prompt(model), images=[IMAGE]):
            projected = model.projector.output.save()
        assert projected.shape[:2] == positions.shape[:2]
        assert torch.equal(projected[valid.to(projected.device)], clean["features"])
        assert clean["features"].shape[0] == int(valid.sum()) == int(clean["mask"].sum())

    def test_reading_the_embedder_first_still_serves_the_scatter(self, model, clean):
        """The scatter is an operation of the wrapper model's forward, instrumented at build, so it is served after the embedder's values."""
        with model.trace(image_prompt(model), images=[IMAGE]):
            model.vision.patch_embeddings.save()
            model.projector.output.save()
            features = model.vision.image_features.save()
        assert torch.equal(features, clean["features"])


def test_the_12b_names_bind_on_meta():
    """``google/gemma-4-12B``'s own config (no weights cached), built on meta: the names bind, the sizes are its embedder's."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained("google/gemma-4-12B")
    model = StandardizedTransformer("google/gemma-4-12B", tokenizer=StandardizedTransformer(TEXT_REPO).tokenizer)
    assert model.family is gemma4_unified_text and type(model._module).__name__ == "Gemma4UnifiedForConditionalGeneration"
    assert model.vision is model.get("model.embed_vision") and model.projector is model.get("model.embed_vision.multimodal_embedder")
    assert (model.vision.hidden_size, model.vision.patch_size) == (3840, 48)
    assert model.vision.patch_embed._module.in_features == 48 * 48 * 3
    assert model.projector._module.embedding_projection.out_features == model.hidden_size == config.text_config.hidden_size
    assert model.vision.support() == {} and not any(name.startswith("vision.") for name in model.support())
