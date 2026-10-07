"""OLMo 2 on vLLM's transformers backend, against the transformers engine: transformers' modules, post-norm."""

from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import olmo2


class TestVLLMOlmo2(VLLMFamilySuite):
    REPO = "allenai/OLMo-2-0425-1B"
    FAMILY = olmo2
    NATIVE = LLAMA_ROWS
    MEMORY = 0.3
