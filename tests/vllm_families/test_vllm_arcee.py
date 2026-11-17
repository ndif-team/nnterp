"""Arcee (AFM) on vLLM's transformers backend, against the transformers engine: transformers' modules, an ungated MLP."""

from vllm_suite import LLAMA_ROWS, NO_ROTARY, VLLMFamilySuite

from nnterp.families.vllm import arcee


class TestVLLMArcee(VLLMFamilySuite):
    REPO = "arcee-ai/AFM-4.5B"
    FAMILY = arcee
    UNAVAILABLE = (VLLMFamilySuite.UNAVAILABLE | NO_ROTARY) - {"mlp.mlp_activation"}  # the transformers backend's rotary is not served
    NATIVE = LLAMA_ROWS
    MEMORY = 0.45
