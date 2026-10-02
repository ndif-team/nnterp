"""Qwen3 on vLLM, against the transformers engine."""

from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import qwen3


class TestVLLMQwen3(VLLMFamilySuite):
    REPO = "Qwen/Qwen3-8B"
    FAMILY = qwen3
    NATIVE = LLAMA_ROWS
    MEMORY = 0.6
