"""GLM-4, end to end: a sandwich block whose post_attention_layernorm is the pre-MLP norm."""

import torch
from suite import FamilySuite, LLAMA_ROWS, PROMPT

from nnter.families import glm4


class TestGlm4(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-Glm4ForCausalLM"
    FAMILY = glm4
    NATIVE = LLAMA_ROWS

    def test_contributions_are_the_post_norms(self, model):
        with model.trace(PROMPT):
            attn = model.layers[0].self_attn.attention_output.save()
            post_attn = model.layers[0].post_self_attn_layernorm.output.save()
            mlp = model.layers[0].mlp.mlp_output.save()
            post_mlp = model.layers[0].post_mlp_layernorm.output.save()
        assert torch.equal(attn, post_attn) and torch.equal(mlp, post_mlp)
