"""StarCoder2, end to end."""

from suite import FamilySuite, LLAMA_ROWS

from nnterp.families import starcoder2


class TestStarcoder2(FamilySuite):
    REPO = "hf-internal-testing/tiny-random-Starcoder2ForCausalLM"
    FAMILY = starcoder2
    NATIVE = LLAMA_ROWS
