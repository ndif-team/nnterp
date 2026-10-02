"""GLM-4-MoE, end to end: a dense first block, then a mixture of experts with a shared expert."""

import torch
from suite import FamilySuite, LLAMA_ROWS, PROMPT

from nnterp.families import glm4_moe


class TestGlm4Moe(FamilySuite):
    REPO = "trl-internal-testing/tiny-Glm4MoeForCausalLM"
    FAMILY = glm4_moe
    NATIVE = LLAMA_ROWS

    def test_dense_then_mixture(self, model):
        dense = model.config.first_k_dense_replace
        kinds = [type(layer.mlp._module).__name__ for layer in model.layers]
        assert kinds == ["Glm4MoeMLP"] * dense + ["Glm4MoeMoE"] * (len(kinds) - dense)

    def test_head_dim_is_the_configs(self, model):
        assert model.head_dim == model.config.head_dim

    def test_mlp_output_includes_the_shared_expert(self, model):
        layer = model.layers[model.config.first_k_dense_replace]
        with model.trace(PROMPT):
            routed = layer.mlp.experts.output.save()
            shared = layer.mlp.shared_experts.output.save()
            out = layer.mlp.mlp_output.save()
        assert torch.allclose(out, routed.view(out.shape) + shared)
