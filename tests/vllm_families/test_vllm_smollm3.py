"""SmolLM3 on vLLM's transformers backend, against the transformers engine: every fourth block without rotary embeddings."""

from vllm_suite import LLAMA_ROWS, NO_ROTARY, VLLMFamilySuite

from nnterp.families.vllm import smollm3


class TestVLLMSmolLM3(VLLMFamilySuite):
    REPO = "HuggingFaceTB/SmolLM3-3B"
    FAMILY = smollm3
    NATIVE = LLAMA_ROWS
    MEMORY = 0.4
    UNAVAILABLE = VLLMFamilySuite.UNAVAILABLE | NO_ROTARY  # the transformers backend; every fourth block is NoPE besides
