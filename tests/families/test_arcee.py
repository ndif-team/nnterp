"""Arcee, end to end."""

from suite import FamilySuite, LLAMA_ROWS

from nnterp.families import arcee


class TestArcee(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-ArceeForCausalLM"
    FAMILY = arcee
    NATIVE = LLAMA_ROWS
    NEURONS_FROM_ACTIVATION = False          # relu squared: down_proj takes act(up)**2
