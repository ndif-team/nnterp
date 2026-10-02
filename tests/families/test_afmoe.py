"""AFMoE, end to end: a sandwich block, a dense first block, then a mixture with a shared expert."""

import pytest
import torch
from suite import FamilySuite, LLAMA_ROWS, PROMPT

from nnter import Unavailable
from nnter.families import afmoe


class TestAfmoe(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-AfmoeForCausalLM"
    FAMILY = afmoe
    NATIVE = LLAMA_ROWS
    MLP_NORM = "pre_mlp_layernorm"           # sandwich: post_attention_layernorm follows the attention

    def test_contributions_are_the_post_norms(self, model):
        with model.trace(PROMPT):
            attn = model.layers[0].self_attn.attention_output.save()
            post_attn = model.layers[0].post_attention_layernorm.output.save()
            mlp = model.layers[0].mlp.mlp_output.save()
            post_mlp = model.layers[0].post_mlp_layernorm.output.save()
        assert torch.equal(attn, post_attn) and torch.equal(mlp, post_mlp)

    def test_dense_then_mixture(self, model):
        dense = model.config.num_dense_layers
        kinds = [type(layer.mlp._module).__name__ for layer in model.layers]
        assert kinds == ["AfmoeMLP"] * dense + ["AfmoeSparseMoeBlock"] * (len(kinds) - dense)

    def test_mixture_output_includes_the_shared_expert(self, model):
        layer = model.layers[model.config.num_dense_layers]
        with model.trace(PROMPT):
            shared = layer.mlp.shared_experts.output.save()
            routed = layer.mlp.experts.output.save()
            out = layer.mlp.output.save()
        torch.testing.assert_close(out, shared + routed.view(out.shape))

    def test_shared_expert_has_no_contribution(self, model):
        shared = model.layers[model.config.num_dense_layers].mlp.shared_experts
        assert "shared expert" in shared.support()["mlp_output"]
        with pytest.raises(Unavailable, match="shared expert"):
            with model.trace(PROMPT):
                shared.mlp_output.save()
