"""Nemotron on vLLM, against the transformers engine."""

from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import nemotron


class TestVLLMNemotron(VLLMFamilySuite):
    REPO = "nvidia/Nemotron-Mini-4B-Instruct"
    FAMILY = nemotron
    NATIVE = LLAMA_ROWS
    MEMORY = 0.4
