"""HyperCLOVA X, end to end: a sandwich block whose post-norms' outputs are added times residual_multiplier; and
HyperCLOVA X Vision V2's Qwen2.5-VL tower and linear projector (transformers 5.18 and later)."""

import pytest
import test_granite
import torch
from qwen_vision_suite import PER_IMAGE_UNAVAILABLE, WIDE, QwenVisionSuite
from suite import FamilySuite, LLAMA_ROWS, PROMPT, rows
from vision_suite import IMAGE, align_processor, image_prompt

from nnterp.components import QwenVision, QwenVisionAttention
from nnterp.families import hyperclovax

REPO = "hf-tiny-v2/tiny-random-HyperCLOVAXForCausalLM"
VISION_REPO = "hf-tiny-v2/tiny-random-HyperCLOVAXVisionV2ForConditionalGeneration"
needs_vision_wrapper = pytest.mark.skipif(
    hyperclovax.HyperCLOVAXVisionV2Model is None, reason="HyperCLOVA X Vision V2 is in transformers 5.18 and later"
)


class TestHyperCLOVAX(FamilySuite):
    REPO = REPO
    FAMILY = hyperclovax
    NATIVE = LLAMA_ROWS

    def test_contributions_are_the_post_norms(self, model):
        with model.trace(PROMPT):
            attn = model.layers[0].self_attn.attention_output.save()
            post_attn = model.layers[0].post_norm1.output.save()
            mlp = model.layers[0].mlp.mlp_output.save()
            post_mlp = model.layers[0].post_norm2.output.save()
        assert torch.equal(attn, post_attn) and torch.equal(mlp, post_mlp)


class TestHyperCLOVAXScaled(test_granite.TestGraniteScaled):
    """The same weights with every multiplier away from 1.0 (Granite's rewrite); here ``logits_scaling`` multiplies."""

    REPO = test_granite._scaled_checkpoint(REPO)
    FAMILY = hyperclovax
    NATIVE = LLAMA_ROWS

    def test_contributions_are_the_scaled_module_outputs(self, model):
        with model.trace(PROMPT):
            post_attn = model.layers[0].post_norm1.output.save()
            attn = model.layers[0].self_attn.attention_output.save()
            post_mlp = model.layers[0].post_norm2.output.save()
            mlp = model.layers[0].mlp.mlp_output.save()
        torch.testing.assert_close(attn, post_attn * 0.22)
        torch.testing.assert_close(mlp, post_mlp * 0.22)

    def test_logits_are_the_head_over_logits_scaling(self, model):
        """Granite's test, inverted: HyperCLOVA X multiplies the head's output by ``logits_scaling``."""
        with model.trace(PROMPT):
            resid = model.layers[-1].layer_output.save()
            raw = model.lm_head.output.save()
            logits = model.logits.save()
        torch.testing.assert_close(logits, raw * 4.0)
        torch.testing.assert_close(model.project_on_vocab(resid), logits)


class TestHyperCLOVAXWithoutPostNorm(FamilySuite):
    """``use_post_norm`` off: the post-norms are identities, so the contributions are the modules' outputs times the multiplier."""

    REPO = test_granite._scaled_checkpoint(REPO, use_post_norm=False)
    FAMILY = hyperclovax
    NATIVE = LLAMA_ROWS

    def test_contributions_are_the_scaled_module_outputs(self, model):
        with model.trace(PROMPT):
            attn_raw = model.layers[0].self_attn.output[0].save()
            attn = model.layers[0].self_attn.attention_output.save()
            mlp_raw = model.layers[0].mlp.output.save()
            mlp = model.layers[0].mlp.mlp_output.save()
        torch.testing.assert_close(attn, attn_raw * 0.22)
        torch.testing.assert_close(mlp, mlp_raw * 0.22)


def align_vision_processor(model):
    """The tiny checkpoint's processor merges 2x2 blocks where its tower's merger takes 1x1, and its image token is not
    the config's: set both to the model's."""
    align_processor(model, image_processor={"merge_size": model.config.vision_config.spatial_merge_size})


@needs_vision_wrapper
class TestHyperCLOVAXVisionWrapper(TestHyperCLOVAX):
    """HyperCLOVA X Vision V2 loaded as the wrapper with its processor: the text stack at ``model.language_model``."""

    REPO = VISION_REPO
    NATIVE = rows("model.language_model", "layers", "embed_tokens", "norm")
    LOAD_KWARGS = {"task": "image-text-to-text"}
    EXPECTED_UNAVAILABLE = PER_IMAGE_UNAVAILABLE


@needs_vision_wrapper
class TestHyperCLOVAXVision(QwenVisionSuite):
    """The Qwen2.5-VL ViT at ``model.vision_model`` (its merger inside) and the linear ``model.projector`` after it."""

    REPO = VISION_REPO
    FAMILY = hyperclovax
    TEXT_REPO = TestHyperCLOVAX.REPO
    WINDOWED = True
    VISION_NATIVE = {
        **{key: value.replace("model.visual", "model.vision_model") for key, value in QwenVisionSuite.VISION_NATIVE.items()},
        "projector": "model.projector",
    }
    fix_processor = staticmethod(align_vision_processor)

    def test_the_tower_is_packed(self, model):
        """The Qwen ViT as on Qwen2.5-VL, but the projector is the linear layer after the tower, not its merger."""
        assert isinstance(model.vision, QwenVision)
        assert all(isinstance(layer.self_attn, QwenVisionAttention) for layer in model.vision.layers)
        assert model.projector._module is model.get("model.projector")._module
        assert model.vision.merger._module is model.get("model.vision_model.merger")._module

    def test_the_merger_output_is_in_scatter_order_unless_windowed(self, model):
        """The merger's output is in window order; the tower restores the order, so the projector's is the scatter's."""
        with model.trace(image_prompt(model), images=[WIDE]):
            mask = model.vision.image_token_mask.save()
            merged = model.vision.merger.output.save()
            projected = model.projector.output.save()
            features = model.vision.image_features.save()
            first = model.layers[0].input.save()
        assert torch.equal(first[mask], features)
        assert torch.equal(projected, features)
        assert merged.shape[0] == features.shape[0]
        torch.testing.assert_close(model.projector._module(merged).sort(0).values, features.sort(0).values)
