"""OLMoE on vLLM, against the transformers engine: a mixture of experts for an MLP."""

from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import olmoe


class TestVLLMOlmoe(VLLMFamilySuite):
    REPO = "hf-internal-testing/tiny-random-OlmoeForCausalLM"
    FAMILY = olmoe
    NATIVE = LLAMA_ROWS
