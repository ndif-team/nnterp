"""Qwen2, end to end."""

from suite import FamilySuite, LLAMA_ROWS

from nnter.families import qwen2


class TestQwen2(FamilySuite):
    REPO = "yujiepan/qwen2-tiny-random"
    FAMILY = qwen2
    NATIVE = LLAMA_ROWS
