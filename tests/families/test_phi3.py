"""Phi-3, end to end."""

from suite import FamilySuite, LLAMA_ROWS

from nnterp.families import phi3


class TestPhi3(FamilySuite):
    REPO = "katuni4ka/tiny-random-phi3"
    FAMILY = phi3
    NATIVE = LLAMA_ROWS
