"""BitNet b1.58, end to end."""

from suite import FamilySuite, LLAMA_ROWS

from nnterp.families import bitnet


class TestBitnet(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-BitNetForCausalLM"
    FAMILY = bitnet
    NATIVE = LLAMA_ROWS
    PREMIX_IS_HEAD_OUTPUTS = False           # attn_sub_norm between the heads and o_proj
    NEURONS_FROM_ACTIVATION = False          # ffn_sub_norm between act(gate)*up and down_proj
