"""Phi-3, end to end."""

from suite import FamilySuite, LLAMA_ROWS

from nnter.families import phi3


class TestPhi3(FamilySuite):
    REPO = "trl-internal-testing/tiny-Phi3ForCausalLM"
    FAMILY = phi3
    NATIVE = LLAMA_ROWS
