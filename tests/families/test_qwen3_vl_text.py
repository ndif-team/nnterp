"""Qwen3-VL text, end to end: Qwen3's block under interleaved M-RoPE, DeepStack after the first blocks; and the packed Qwen ViT."""

from qwen_vision_suite import PER_IMAGE_UNAVAILABLE, DeepstackSuite, MRopeSuite, QwenVisionSuite
from suite import FamilySuite, rows

from nnterp.families import qwen3_vl_text


class TestQwen3VL(DeepstackSuite, MRopeSuite, FamilySuite):
    """The wrapper with its processor (there is no text-only class): the text stack at ``model.language_model``."""

    REPO = "yujiepan/qwen3-vl-tiny-random"
    FAMILY = qwen3_vl_text
    NATIVE = rows("model.language_model", "layers", "embed_tokens", "norm")
    LOAD_KWARGS = {"task": "image-text-to-text"}
    EXPECTED_UNAVAILABLE = PER_IMAGE_UNAVAILABLE


class TestQwen3VLVision(QwenVisionSuite):
    """Qwen3-VL's ViT: a learned ``pos_embed`` after ``patch_embed``, and three deepstack taps."""

    REPO = TestQwen3VL.REPO
    FAMILY = qwen3_vl_text
