"""Qwen3-MoE, end to end."""

from suite import FamilySuite, LLAMA_ROWS

from nnterp.families import qwen3_moe


class TestQwen3Moe(FamilySuite):
    REPO = "trl-internal-testing/tiny-Qwen3MoeForCausalLM"
    FAMILY = qwen3_moe
    NATIVE = LLAMA_ROWS
    MLP_WIDTH_KEY = "moe_intermediate_size"  # every block is a mixture of experts; ``intermediate_size`` is the unused dense width
