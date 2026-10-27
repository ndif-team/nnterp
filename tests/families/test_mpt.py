"""MPT, end to end: residual added inside the MLP."""

import torch
from suite import FamilySuite, rows, PROMPT

from nnterp.families import mpt


class TestMpt(FamilySuite):
    REPO = "hf-internal-testing/tiny-random-MptForCausalLM"
    FAMILY = mpt
    NATIVE = rows("transformer", "blocks", "wte", "norm_f", attn="attn", mlp="ffn", ln1="norm_1", ln2="norm_2")
    REFUSES_IN_PLACE_QKV = True  # q/k/v come out of one chunk()

    def test_mlp_contribution_is_before_the_residual_add(self, model):
        with model.trace(PROMPT):
            mlp = model.layers[0].mlp.mlp_output.save()
            mlp_module = model.layers[0].mlp.output.save()
        assert not torch.equal(mlp, mlp_module)
