"""OLMo 3 on vLLM, against the transformers engine: a post-norm block that takes and returns the stream."""

from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import olmo3


class TestVLLMOlmo3(VLLMFamilySuite):
    REPO = "allenai/Olmo-3-7B-Instruct"
    FAMILY = olmo3
    NATIVE = LLAMA_ROWS
    MEMORY = 0.55
