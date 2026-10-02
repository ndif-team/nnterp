"""ERNIE 4.5 dense, end to end."""

from suite import FamilySuite, LLAMA_ROWS

from nnter.families import ernie4_5


class TestErnie45(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-Ernie4_5ForCausalLM"
    FAMILY = ernie4_5
    NATIVE = LLAMA_ROWS
