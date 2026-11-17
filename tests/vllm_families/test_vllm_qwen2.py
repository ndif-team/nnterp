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
    # That difference is 3.6e-3 of the logits' scale, and halving the middle block's layer_mid raises it to 5.2e-3:
    # the same edit doubles the difference between transformers' own eager and sdpa kernels (4.4e-6 to 8.6e-6), so
    # the halved logits are held at twice the default. layer_mid itself agrees to 9e-6.
    MODULE_TOLERANCE = {"layer_mid": 1e-2}
