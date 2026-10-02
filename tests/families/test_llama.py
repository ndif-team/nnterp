"""Llama, end to end: the family the vocabulary is taken from."""

from suite import FamilySuite, LLAMA_ROWS

from nnterp.families import llama


class TestLlama(FamilySuite):
    REPO = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    FAMILY = llama
    NATIVE = LLAMA_ROWS
