"""Helium, end to end."""

from suite import FamilySuite, LLAMA_ROWS

from nnterp.families import helium


class TestHelium(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-HeliumForCausalLM"
    FAMILY = helium
    NATIVE = LLAMA_ROWS
