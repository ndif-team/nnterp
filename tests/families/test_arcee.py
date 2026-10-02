"""Arcee, end to end."""

from suite import FamilySuite, LLAMA_ROWS

from nnter.families import arcee


class TestArcee(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-ArceeForCausalLM"
    FAMILY = arcee
    NATIVE = LLAMA_ROWS
