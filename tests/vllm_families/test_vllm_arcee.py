"""Arcee (AFM) on vLLM, against the transformers engine."""

from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import arcee


class TestVLLMArcee(VLLMFamilySuite):
    REPO = "arcee-ai/AFM-4.5B"
    FAMILY = arcee
    NATIVE = LLAMA_ROWS
    MEMORY = 0.45
