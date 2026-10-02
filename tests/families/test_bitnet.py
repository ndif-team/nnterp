"""BitNet b1.58, end to end."""

from suite import FamilySuite, LLAMA_ROWS

from nnterp.families import bitnet


class TestBitnet(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-BitNetForCausalLM"
    FAMILY = bitnet
    NATIVE = LLAMA_ROWS
