"""ERNIE 4.5 dense on vLLM, against the transformers engine: vLLM's Llama classes."""

from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import ernie4_5


class TestVLLMErnie4_5(VLLMFamilySuite):
    REPO = "baidu/ERNIE-4.5-0.3B-PT"
    FAMILY = ernie4_5
    NATIVE = LLAMA_ROWS
