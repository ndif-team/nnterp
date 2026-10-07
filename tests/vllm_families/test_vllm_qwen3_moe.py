"""Qwen3-MoE on vLLM, against the transformers engine: a mixture of experts for an MLP."""

from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import qwen3_moe


class TestVLLMQwen3Moe(VLLMFamilySuite):
    REPO = "trl-internal-testing/tiny-Qwen3MoeForCausalLM"
    FAMILY = qwen3_moe
    NATIVE = LLAMA_ROWS
