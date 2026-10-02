"""OLMo 1, end to end."""

from suite import FamilySuite, LLAMA_ROWS

from nnter.families import olmo


class TestOlmo(FamilySuite):
    REPO = "katuni4ka/tiny-random-olmo-hf"
    FAMILY = olmo
    NATIVE = LLAMA_ROWS
