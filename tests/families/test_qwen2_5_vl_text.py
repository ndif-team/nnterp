"""Qwen2.5-VL text, end to end: Qwen2's block under M-RoPE, inside the wrapper; and the packed, windowed Qwen ViT."""

from qwen_vision_suite import PER_IMAGE_UNAVAILABLE, MRopeSuite, QwenVisionSuite
from suite import FamilySuite, rows

from nnterp.families import qwen2_5_vl_text


class TestQwen2_5VL(MRopeSuite, FamilySuite):
    """The wrapper with its processor (there is no text-only class): the text stack at ``model.language_model``."""

    REPO = "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration"
    FAMILY = qwen2_5_vl_text
    NATIVE = rows("model.language_model", "layers", "embed_tokens", "norm")
    LOAD_KWARGS = {"task": "image-text-to-text"}
    EXPECTED_UNAVAILABLE = PER_IMAGE_UNAVAILABLE


class TestQwen2_5VLVision(QwenVisionSuite):
    """Qwen2.5-VL's ViT: RMSNorm blocks, window attention, the patches held in window order between entry and the merger."""

    REPO = TestQwen2_5VL.REPO
    FAMILY = qwen2_5_vl_text
    WINDOWED = True
