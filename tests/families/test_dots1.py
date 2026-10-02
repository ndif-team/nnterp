"""dots.llm1, end to end."""

from suite import FamilySuite, LLAMA_ROWS

from nnter.families import dots1


class TestDots1(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-Dots1ForCausalLM"
    FAMILY = dots1
    NATIVE = LLAMA_ROWS
    MLP_WIDTH_KEY = "moe_intermediate_size"  # every block of the tiny checkpoint is a mixture of experts
