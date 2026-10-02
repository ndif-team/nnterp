"""Nemotron, end to end."""

from suite import FamilySuite, LLAMA_ROWS

from nnter.families import nemotron


class TestNemotron(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-NemotronForCausalLM"
    FAMILY = nemotron
    NATIVE = LLAMA_ROWS
