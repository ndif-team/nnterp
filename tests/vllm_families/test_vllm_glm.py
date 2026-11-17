"""GLM-4 (the first generation) on vLLM's transformers backend, against the transformers engine: transformers' modules, Llama's block."""

from vllm_suite import LLAMA_ROWS, NO_ROTARY, VLLMFamilySuite

from nnterp.families.vllm import glm


class TestVLLMGlm(VLLMFamilySuite):
    REPO = "zai-org/glm-edge-1.5b-chat"
    FAMILY = glm
    UNAVAILABLE = VLLMFamilySuite.UNAVAILABLE | NO_ROTARY  # the transformers backend's rotary is not served
    NATIVE = LLAMA_ROWS
    MEMORY = 0.3
