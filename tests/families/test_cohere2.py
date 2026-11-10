"""Cohere 2 (Command-R7B), end to end: a parallel block, sliding layers with rotary, full layers without."""

import glob
import json
import os
import tempfile

import pytest

import torch
from suite import NO_ROTARY, FamilySuite, PROMPT, rows
from vision_suite import VisionSuite, WrapperSuite, align_processor, siglip_rows, wrapper_of

from nnterp.families import cohere, cohere2


class TestCohere2(FamilySuite):
    REPO = "trl-internal-testing/tiny-Cohere2ForCausalLM"
    FAMILY = cohere2
    NATIVE = rows("model", "layers", "embed_tokens", "norm", ln2=None)
    MLP_NORM = "input_layernorm"  # parallel: one norm feeds both sublayers
    EXPECTED_UNAVAILABLE = {**NO_ROTARY, "layer_mid": "a parallel block"}  # the full-attention blocks have no rotary

    def test_both_layer_types_are_covered(self, model):
        assert set(model.config.layer_types) == {"sliding_attention", "full_attention"}

    def test_full_layers_take_the_unrotated_projections(self, model):
        """Only sliding layers apply the rotary: on a full layer the queries are ``q_proj``'s output, heads first."""
        kinds = model.config.layer_types
        full, sliding = kinds.index("full_attention"), kinds.index("sliding_attention")
        projected = {}
        for i in (full, sliding):
            model.layers[i].self_attn.source  # nnsight instruments a forward on the first `.source` access; a child's output read before that in the same trace leaves the call uninstrumented
            with model.trace(PROMPT):
                attn = model.layers[i].self_attn
                q_proj = attn.q_proj.output.save()
                queries = attn.attention_queries.save()
            projected[i] = (q_proj.view(*q_proj.shape[:2], model.num_heads, model.head_dim).transpose(1, 2), queries)
        torch.testing.assert_close(*projected[full])
        assert not torch.allclose(*projected[sliding])

    def test_logits_scale_is_cohere_s(self, model):
        assert model.family.project_on_vocab is cohere.project_on_vocab


def _scaled_checkpoint(repo=TestCohere2.REPO, logit_scale=0.25):
    """The tiny checkpoint with Command-R7B's ``logit_scale`` (0.25); the tiny config's is 1.0. Only ``config.json`` is rewritten."""
    snapshot = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*"))[0]
    patched = tempfile.mkdtemp(prefix="cohere2-scaled-")
    for name in os.listdir(snapshot):
        if name != "config.json":
            os.symlink(os.path.realpath(os.path.join(snapshot, name)), os.path.join(patched, name))
    config = json.load(open(os.path.join(snapshot, "config.json")))
    config["logit_scale"] = logit_scale
    json.dump(config, open(os.path.join(patched, "config.json"), "w"))
    return patched


class TestCohere2Scaled(TestCohere2):
    """The same weights with a ``logit_scale`` away from 1.0: ``logits`` is the head's output times it, and so is the lens."""

    REPO = _scaled_checkpoint()

    def test_logits_are_the_scaled_head_output(self, model):
        with model.trace(PROMPT):
            raw = model.lm_head_output.save()
            logits = model.logits.save()
        assert model.config.logit_scale == 0.25
        torch.testing.assert_close(logits, raw * 0.25)


class TestAyaVisionWrapper(WrapperSuite):
    """Aya Vision around Cohere 2, built from the tiny text checkpoint's config (no tiny wrapper checkpoint exists): the text stack at
    ``model.language_model``. Its text config carries Command-R7B's ``logit_scale``, which the wrapper never applies."""

    FAMILY = cohere2

    @pytest.fixture(scope="class")
    def model(self):
        return wrapper_of(
            TestCohere2.REPO, "AyaVisionConfig", dict(hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32),
            text_overrides=dict(logit_scale=0.25),
        )

    def test_logits_are_the_unscaled_head_output(self, model):
        """``AyaVisionForConditionalGeneration`` returns ``lm_head``'s output as it is, though its ``text_config`` sets a scale."""
        with model.trace(PROMPT):
            raw = model.lm_head_output.save()
            logits = model.logits.save()
        assert model.config.text_config.logit_scale == 0.25
        assert torch.equal(logits, raw)


class TestAyaVisionVision(VisionSuite):
    """Aya Vision's SigLIP tower and pixel-shuffling projector, and the tower's image values."""

    REPO = "hf-tiny-v2/tiny-random-AyaVisionForConditionalGeneration"
    FAMILY = cohere2
    TEXT_REPO = TestCohere2.REPO
    VISION_NATIVE = siglip_rows()

    @staticmethod
    def fix_processor(model):
        """The processor counts image tokens with Aya Vision 8B's 364-pixel image and 28-pixel merged patch; the tiny
        tower takes 64 pixels in 8-pixel patches merged 2x2."""
        vision = model.config.vision_config
        align_processor(model, img_size=vision.image_size, patch_size=vision.patch_size * model.config.downsample_factor)


class TestCohere2VisionVision(TestAyaVisionVision):
    """Cohere2-Vision (Command-A Vision): Aya Vision's layout."""

    REPO = "hf-tiny-v2/tiny-random-Cohere2VisionForConditionalGeneration"

    @staticmethod
    def fix_processor(model):
        """The processor's ``patch_size`` is the side of a tile's token grid: 16 where the tiny tower's 64-pixel tile in
        8-pixel patches merged 2x2 makes 4."""
        vision = model.config.vision_config
        align_processor(model, patch_size=vision.image_size // (vision.patch_size * model.config.downsample_factor))
