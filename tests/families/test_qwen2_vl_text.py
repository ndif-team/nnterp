"""Qwen2-VL text, end to end: Qwen2's block under M-RoPE, inside the wrapper; and the packed Qwen ViT."""

from qwen_vision_suite import PACKED_UNAVAILABLE, MRopeSuite, QwenVisionSuite
from suite import FamilySuite, rows

from nnterp.families import qwen2_vl_text


class TestQwen2VL(MRopeSuite, FamilySuite):
    """The wrapper with its processor (there is no text-only class): the text stack at ``model.language_model``."""

    REPO = "yujiepan/qwen2-vl-tiny-random"
    FAMILY = qwen2_vl_text
    NATIVE = rows("model.language_model", "layers", "embed_tokens", "norm")
    LOAD_KWARGS = {"task": "image-text-to-text"}
    EXPECTED_UNAVAILABLE = PACKED_UNAVAILABLE


class TestQwen2VLVision(QwenVisionSuite):
    """Qwen2-VL's ViT: LayerNorm blocks, the merger inside the tower, ``embed_dim`` its width."""

    REPO = TestQwen2VL.REPO
    FAMILY = qwen2_vl_text
