"""Phi-3.5-MoE, end to end."""

from suite import FamilySuite, LLAMA_ROWS

from nnterp.families import phimoe


class TestPhimoe(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-PhimoeForCausalLM"
    FAMILY = phimoe
    NATIVE = LLAMA_ROWS
