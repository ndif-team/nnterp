"""Qwen2, end to end."""

from suite import FamilySuite, LLAMA_ROWS, rows
from vision_suite import IMAGE, VisionSuite, align_processor, image_prompt, siglip_rows

from nnterp.families import qwen2


class TestQwen2(FamilySuite):
    REPO = "yujiepan/qwen2-tiny-random"
    FAMILY = qwen2
    NATIVE = LLAMA_ROWS


class TestLlavaInterleaveWrapper(TestQwen2):
    """llava-interleave (Llava around Qwen2): the text stack at ``model.language_model``."""

    REPO = "llava-hf/llava-interleave-qwen-0.5b-hf"
    NATIVE = rows("model.language_model", "layers", "embed_tokens", "norm")
    LOAD_KWARGS = {"task": "image-text-to-text"}


class TestLlavaInterleaveVision(VisionSuite):
    """llava-interleave: Llava around Qwen2 with a SigLIP tower, real weights."""

    REPO = "llava-hf/llava-interleave-qwen-0.5b-hf"
    FAMILY = qwen2
    TEXT_REPO = TestQwen2.REPO
    VISION_NATIVE = siglip_rows()


class TestLlavaOnevisionVision(VisionSuite):
    """LLaVA-OneVision: SigLIP over the image's crops; the wrapper unpads the projector's output and adds newline tokens."""

    REPO = "hf-tiny-v2/tiny-random-LlavaOnevisionForConditionalGeneration"
    FAMILY = qwen2
    TEXT_REPO = TestQwen2.REPO
    VISION_NATIVE = siglip_rows()

    @staticmethod
    def fix_processor(model):
        """The processor's token count and crop grid are unset or the 384-pixel default; the tiny tower takes 16 pixels."""
        config, side = model.config, model.config.vision_config.image_size // model.config.vision_config.patch_size
        align_processor(
            model, num_image_tokens=side * side, vision_feature_select_strategy=config.vision_feature_select_strategy,
            image_processor={"image_grid_pinpoints": config.image_grid_pinpoints},
        )

    def test_image_features_are_not_the_projector_output(self, model):
        with model.trace(image_prompt(model), images=[IMAGE]):
            projected = model.projector.output.save()
            features = model.vision.image_features.save()
        assert features.shape[0] != projected.shape[0] * projected.shape[1]
        newline = model.get("model").image_newline.detach()
        assert (features == newline).all(-1).any()
