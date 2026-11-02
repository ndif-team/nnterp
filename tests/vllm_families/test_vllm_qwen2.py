"""Qwen2 on vLLM, against the transformers engine."""

from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import qwen2


class TestVLLMQwen2(VLLMFamilySuite):
    REPO = "Qwen/Qwen2.5-0.5B"
    FAMILY = qwen2
    NATIVE = LLAMA_ROWS
    # Qwen2.5-0.5B's first block has queries near 80 and keys near 130, so its scores are in the thousands and the
    # softmax is as sharp as float32 allows. The queries, keys and values agree with transformers' to 1e-7 of their
    # scale; what the two kernels make of them differs by up to 2.4e-2 of the head outputs' scale there, and the
    # later blocks' interior values carry that (up to 6e-3 of theirs).
    KERNEL_TOLERANCE = 5e-2
