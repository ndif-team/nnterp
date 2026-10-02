"""Gemma 1, end to end."""

from suite import FamilySuite, LLAMA_ROWS

from nnter.families import gemma


class TestGemma(FamilySuite):
    REPO = "trl-internal-testing/tiny-GemmaForCausalLM"
    FAMILY = gemma
    NATIVE = LLAMA_ROWS
