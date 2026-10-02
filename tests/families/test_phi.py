"""Phi, end to end: a parallel block with one norm."""

from suite import FamilySuite, rows

from nnter.families import phi


class TestPhi(FamilySuite):
    REPO = "hf-internal-testing/tiny-random-PhiForCausalLM"
    FAMILY = phi
    NATIVE = rows("model", "layers", "embed_tokens", "final_layernorm", ln2=None)
    MLP_NORM = "input_layernorm"             # parallel: one norm feeds both sublayers
