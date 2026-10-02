"""Mistral on vLLM, against the transformers engine."""

from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import mistral


class TestVLLMMistral(VLLMFamilySuite):
    REPO = "openaccess-ai-collective/tiny-mistral"
    FAMILY = mistral
    NATIVE = LLAMA_ROWS
