"""SmolLM3, end to end."""

from suite import NO_ROTARY, FamilySuite, LLAMA_ROWS

from nnterp.families import smollm3


class TestSmolLM3(FamilySuite):
    REPO = "yujiepan/smollm3-tiny-random"
    FAMILY = smollm3
    NATIVE = LLAMA_ROWS
    EXPECTED_UNAVAILABLE = NO_ROTARY  # block 1 is a NoPE block (no_rope_layers' 0)
