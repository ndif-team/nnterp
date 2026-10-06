"""Qwen2, end to end."""

from suite import FamilySuite, LLAMA_ROWS, rows

from nnterp.families import qwen2


class TestQwen2(FamilySuite):
    REPO = "yujiepan/qwen2-tiny-random"
    FAMILY = qwen2
    NATIVE = LLAMA_ROWS


class TestLlavaInterleaveWrapper(TestQwen2):
    """llava-interleave (Llava around Qwen2, SigLIP tower not named yet): the text stack at ``model.language_model``."""

    REPO = "llava-hf/llava-interleave-qwen-0.5b-hf"
    NATIVE = rows("model.language_model", "layers", "embed_tokens", "norm")
    LOAD_KWARGS = {"task": "image-text-to-text"}
