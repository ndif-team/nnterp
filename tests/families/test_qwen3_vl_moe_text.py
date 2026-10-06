"""Qwen3-VL-MoE text, end to end: Qwen3-VL's block and DeepStack with a sparse mixture of experts; and the packed Qwen ViT."""

from qwen_vision_suite import PACKED_UNAVAILABLE, DeepstackSuite, MRopeSuite, QwenVisionSuite
from suite import FamilySuite, rows

from nnterp.families import qwen3_vl_moe_text


class TestQwen3VLMoe(DeepstackSuite, MRopeSuite, FamilySuite):
    """The wrapper with its processor (there is no text-only class): the text stack at ``model.language_model``."""

    REPO = "yujiepan/qwen3-vl-moe-tiny-random"
    FAMILY = qwen3_vl_moe_text
    NATIVE = rows("model.language_model", "layers", "embed_tokens", "norm")
    LOAD_KWARGS = {"task": "image-text-to-text"}
    EXPECTED_UNAVAILABLE = PACKED_UNAVAILABLE
    MLP_WIDTH_KEY = "moe_intermediate_size"  # every block is a mixture of experts


class TestQwen3VLMoeVision(QwenVisionSuite):
    """Qwen3-VL-MoE's ViT: Qwen3-VL's, under the MoE wrapper's own class names."""

    REPO = TestQwen3VLMoe.REPO
    FAMILY = qwen3_vl_moe_text
