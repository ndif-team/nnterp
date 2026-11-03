"""OLMo on vLLM's transformers backend, against the transformers engine: transformers' modules, Llama's block."""

from vllm_suite import LLAMA_ROWS, NO_ROTARY, VLLMFamilySuite

from nnterp.families.vllm import olmo


class TestVLLMOlmo(VLLMFamilySuite):
    REPO = "allenai/OLMo-1B-hf"
    FAMILY = olmo
    NATIVE = LLAMA_ROWS
    MEMORY = 0.3
    UNAVAILABLE = VLLMFamilySuite.UNAVAILABLE | NO_ROTARY  # the transformers backend's rotary is not served
