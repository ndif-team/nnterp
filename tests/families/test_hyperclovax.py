"""HyperCLOVA X, end to end: a sandwich block whose post-norms' outputs are added times residual_multiplier."""

import test_granite
import torch
from suite import FamilySuite, LLAMA_ROWS, PROMPT

from nnter.families import hyperclovax

REPO = "hf-tiny-v2/tiny-random-HyperCLOVAXForCausalLM"


class TestHyperCLOVAX(FamilySuite):
    REPO = REPO
    FAMILY = hyperclovax
    NATIVE = LLAMA_ROWS

    def test_contributions_are_the_post_norms(self, model):
        with model.trace(PROMPT):
            attn = model.layers[0].self_attn.attention_output.save()
            post_attn = model.layers[0].post_norm1.output.save()
            mlp = model.layers[0].mlp.mlp_output.save()
            post_mlp = model.layers[0].post_norm2.output.save()
        assert torch.equal(attn, post_attn) and torch.equal(mlp, post_mlp)


class TestHyperCLOVAXScaled(test_granite.TestGraniteScaled):
    """The same weights with every multiplier away from 1.0 (Granite's rewrite); here ``logits_scaling`` multiplies."""

    REPO = test_granite._scaled_checkpoint(REPO)
    FAMILY = hyperclovax
    NATIVE = LLAMA_ROWS

    def test_contributions_are_the_scaled_module_outputs(self, model):
        with model.trace(PROMPT):
            post_attn = model.layers[0].post_norm1.output.save()
            attn = model.layers[0].self_attn.attention_output.save()
            post_mlp = model.layers[0].post_norm2.output.save()
            mlp = model.layers[0].mlp.mlp_output.save()
        torch.testing.assert_close(attn, post_attn * 0.22)
        torch.testing.assert_close(mlp, post_mlp * 0.22)

    def test_logits_are_the_head_over_logits_scaling(self, model):
        """Granite's test, inverted: HyperCLOVA X multiplies the head's output by ``logits_scaling``."""
        with model.trace(PROMPT):
            resid = model.layers[-1].layer_output.save()
            raw = model.lm_head.output.save()
            logits = model.logits.save()
        torch.testing.assert_close(logits, raw * 4.0)
        torch.testing.assert_close(model.project_on_vocab(resid), logits)


class TestHyperCLOVAXWithoutPostNorm(FamilySuite):
    """``use_post_norm`` off: the post-norms are identities, so the contributions are the modules' outputs times the multiplier."""

    REPO = test_granite._scaled_checkpoint(REPO, use_post_norm=False)
    FAMILY = hyperclovax
    NATIVE = LLAMA_ROWS

    def test_contributions_are_the_scaled_module_outputs(self, model):
        with model.trace(PROMPT):
            attn_raw = model.layers[0].self_attn.output[0].save()
            attn = model.layers[0].self_attn.attention_output.save()
            mlp_raw = model.layers[0].mlp.output.save()
            mlp = model.layers[0].mlp.mlp_output.save()
        torch.testing.assert_close(attn, attn_raw * 0.22)
        torch.testing.assert_close(mlp, mlp_raw * 0.22)
