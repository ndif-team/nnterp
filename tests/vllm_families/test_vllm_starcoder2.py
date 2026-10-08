"""StarCoder2 on vLLM's transformers backend, against the transformers engine: layer norms and an ungated MLP."""

from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import starcoder2


class TestVLLMStarcoder2(VLLMFamilySuite):
    REPO = "TechxGenus/starcoder2-3b-instruct"
    FAMILY = starcoder2
    NATIVE = LLAMA_ROWS
    MEMORY = 0.4
