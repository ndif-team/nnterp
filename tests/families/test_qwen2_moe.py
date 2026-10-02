"""Qwen2-MoE, end to end."""

from suite import FamilySuite, LLAMA_ROWS

from nnter.families import qwen2_moe


class TestQwen2Moe(FamilySuite):
    REPO = "hf-internal-testing/tiny-random-Qwen2MoeForCausalLM"
    FAMILY = qwen2_moe
    NATIVE = LLAMA_ROWS
