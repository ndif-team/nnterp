"""ERNIE 4.5 MoE, end to end."""

from suite import FamilySuite, LLAMA_ROWS

from nnter.families import ernie4_5_moe


class TestErnie45Moe(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-Ernie4_5_MoeForCausalLM"
    FAMILY = ernie4_5_moe
    NATIVE = LLAMA_ROWS
    MLP_WIDTH_KEY = "moe_intermediate_size"  # every block of the tiny checkpoint is a mixture of experts
