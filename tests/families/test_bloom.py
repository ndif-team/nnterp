"""BLOOM, end to end: residual added inside both sublayers."""

import torch
from suite import FamilySuite, rows, PROMPT

from nnter.families import bloom


class TestBloom(FamilySuite):
    REPO = "hf-internal-testing/tiny-random-BloomForCausalLM"
    FAMILY = bloom
    NATIVE = rows("transformer", "h", "word_embeddings", "ln_f", attn="self_attention")

    def test_contributions_are_before_the_residual_add(self, model):
        """Both sublayers add the residual inside the module; the values are the pre-add tensors."""
        with model.trace(PROMPT):
            x = model.layers[0].input.save()
            attn = model.layers[0].self_attn.attention_output.save()
            attn_module = model.layers[0].self_attn.output.save()
        assert not torch.equal(attn, attn_module[0])
        torch.testing.assert_close(attn_module[0], x + attn)
