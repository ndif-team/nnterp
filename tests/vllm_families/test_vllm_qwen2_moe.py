"""Qwen2-MoE on vLLM, against the transformers engine: a mixture of experts for an MLP."""

from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import qwen2_moe


class TestVLLMQwen2Moe(VLLMFamilySuite):
    REPO = "hf-internal-testing/tiny-random-Qwen2MoeForCausalLM"
    FAMILY = qwen2_moe
    NATIVE = LLAMA_ROWS
