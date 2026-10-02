"""GLM-4-MoE-Lite (GLM-4.7-Flash), end to end: multi-head latent attention, dense then mixture-of-experts blocks."""

import torch
from suite import FamilySuite, LLAMA_ROWS, PROMPT

from nnter.families import glm4_moe_lite


class TestGlm4MoeLite(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-Glm4MoeLiteForCausalLM"
    FAMILY = glm4_moe_lite
    NATIVE = LLAMA_ROWS
    KV_HEADS_EXPANDED = True  # latent attention projects keys and values for every head

    def test_latent_attention_widths(self, model):
        assert model.qk_head_dim == model.config.qk_nope_head_dim + model.config.qk_rope_head_dim
        assert model.head_dim == model.config.v_head_dim
        assert model.config.head_dim == model.config.qk_rope_head_dim != model.head_dim  # the config's alias is the rotary part

    def test_mlp_layer_types(self, model):
        kinds = [type(layer.mlp._module).__name__ for layer in model.layers]
        assert kinds == [{"dense": "Glm4MoeLiteMLP", "sparse": "Glm4MoeLiteMoE"}[t] for t in model.config.mlp_layer_types]

    def test_mlp_output_includes_the_shared_expert(self, model):
        layer = model.layers[model.config.mlp_layer_types.index("sparse")]
        with model.trace(PROMPT):
            routed = layer.mlp.experts.output.save()
            shared = layer.mlp.shared_experts.output.save()
            out = layer.mlp.mlp_output.save()
        assert torch.allclose(out, routed.view(out.shape) + shared)
