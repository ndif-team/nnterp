"""SmolLM3, end to end."""

from suite import FamilySuite, LLAMA_ROWS

from nnter.families import smollm3


class TestSmolLM3(FamilySuite):
    REPO = "yujiepan/smollm3-tiny-random"
    FAMILY = smollm3
    NATIVE = LLAMA_ROWS
