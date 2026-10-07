"""GLM-4-MoE on vLLM, against the transformers engine: partial rotary, attention biases, a mixture of experts with a shared expert for an MLP."""

from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import glm4_moe


class TestVLLMGlm4Moe(VLLMFamilySuite):
    REPO = "PrimeIntellect/glm4-moe-tiny"
    FAMILY = glm4_moe
    NATIVE = LLAMA_ROWS
