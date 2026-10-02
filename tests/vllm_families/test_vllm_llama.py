"""Llama on vLLM, against the transformers engine."""

from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import llama


class TestVLLMLlama(VLLMFamilySuite):
    REPO = "HuggingFaceTB/SmolLM2-135M-Instruct"
    FAMILY = llama
    NATIVE = LLAMA_ROWS
