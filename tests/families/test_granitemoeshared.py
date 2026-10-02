"""GraniteMoE-Shared, end to end: a mixture plus a shared expert, added as one scaled sum."""

import test_granite
import torch
from suite import FamilySuite, LLAMA_ROWS, PROMPT

from nnter.families import granitemoeshared

REPO = "hf-tiny-v2/tiny-random-GraniteMoeSharedForCausalLM"
NATIVE = {**LLAMA_ROWS, "layers.0.mlp": "model.layers.0.block_sparse_moe"}


def scaled_mixture(self, model, host):
    """The block's sum of the mixture and the shared expert: ``mlp_output`` over the multiplier."""
    with model.trace(PROMPT):
        out = host.mlp_output.save()
    return out / model.config.residual_multiplier


class TestGraniteMoeShared(FamilySuite):
    REPO = REPO
    FAMILY = granitemoeshared
    mixture_output = scaled_mixture
    NATIVE = NATIVE

    def test_mlp_output_is_the_mixture_plus_the_shared_expert(self, model):
        with model.trace(PROMPT):
            moe = model.layers[0].mlp.output.save()
            shared = model.layers[0].shared_mlp.output.save()
            out = model.layers[0].mlp.mlp_output.save()
        assert torch.equal(out, moe + shared)


class TestGraniteMoeSharedScaled(test_granite.TestGraniteScaled):
    """The same weights with every multiplier away from 1.0 (Granite's rewrite)."""

    REPO = test_granite._scaled_checkpoint(REPO)
    FAMILY = granitemoeshared
    mixture_output = scaled_mixture
    NATIVE = NATIVE

    def test_contributions_are_the_scaled_module_outputs(self, model):
        with model.trace(PROMPT):
            attn_raw = model.layers[0].self_attn.output[0].save()
            attn = model.layers[0].self_attn.attention_output.save()
            moe = model.layers[0].mlp.output.save()
            shared = model.layers[0].shared_mlp.output.save()
            mlp = model.layers[0].mlp.mlp_output.save()
        torch.testing.assert_close(attn, attn_raw * 0.22)
        torch.testing.assert_close(mlp, (moe + shared) * 0.22)


class TestGraniteMoeSharedWithoutSharedExpert(FamilySuite):
    """``shared_intermediate_size`` 0: no shared expert, so the block adds the mixture alone."""

    REPO = test_granite._scaled_checkpoint(REPO, shared_intermediate_size=0)
    FAMILY = granitemoeshared
    mixture_output = scaled_mixture
    MOE_UNAVAILABLE = {"shared_expert_output": "no shared expert"}
    NATIVE = NATIVE

    def test_no_shared_expert(self, model):
        assert all(layer._module.shared_mlp is None for layer in model.layers)
        with model.trace(PROMPT):
            moe = model.layers[0].mlp.output.save()
            out = model.layers[0].mlp.mlp_output.save()
        torch.testing.assert_close(out, moe * 0.22)
