"""Ministral 3, end to end."""

from suite import FamilySuite, LLAMA_ROWS

from nnter.families import ministral3


class TestMinistral3(FamilySuite):
    REPO = "yujiepan/ministral-3-tiny-random"
    FAMILY = ministral3
    NATIVE = LLAMA_ROWS
