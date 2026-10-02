"""Mistral, end to end."""

from suite import FamilySuite, LLAMA_ROWS

from nnterp.families import mistral


class TestMistral(FamilySuite):
    REPO = "hf-internal-testing/tiny-random-MistralForCausalLM"
    FAMILY = mistral
    NATIVE = LLAMA_ROWS
