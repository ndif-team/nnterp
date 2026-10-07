"""ERNIE 4.5 MoE on vLLM, against the transformers engine: a mixture of experts with shared experts for an MLP."""

from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import ernie4_5_moe


class TestVLLMErnie45Moe(VLLMFamilySuite):
    REPO = "yujiepan/ernie-4.5-moe-tiny-random"
    FAMILY = ernie4_5_moe
    NATIVE = LLAMA_ROWS
