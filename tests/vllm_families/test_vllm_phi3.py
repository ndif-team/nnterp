"""Phi-3 on vLLM, against the transformers engine: vLLM's Llama classes."""

from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import phi3


class TestVLLMPhi3(VLLMFamilySuite):
    REPO = "microsoft/Phi-3-mini-4k-instruct"
    FAMILY = phi3
    NATIVE = LLAMA_ROWS
    MEMORY = 0.4
