"""FlexOlmo, end to end: OLMo-2's post-norms around a mixture of experts."""

import torch
from suite import FamilySuite, rows, PROMPT

from nnter.families import flex_olmo


class TestFlexOlmo(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-FlexOlmoForCausalLM"
    FAMILY = flex_olmo
    NATIVE = rows("model", "layers", "embed_tokens", "norm", ln1=None)
    ATTENTION_NORM = None                    # post-norms only: the block input enters the attention
    MLP_NORM = None                          # ... and the MLP takes the residual stream after the attention add

    def test_contributions_are_the_post_norms(self, model):
        with model.trace(PROMPT):
            attn = model.layers[0].self_attn.attention_output.save()
            post_attn = model.layers[0].post_attention_layernorm.output.save()
            mlp = model.layers[0].mlp.mlp_output.save()
            post_ff = model.layers[0].post_feedforward_layernorm.output.save()
        assert torch.equal(attn, post_attn) and torch.equal(mlp, post_ff)

    def test_every_mlp_is_a_mixture(self, model):
        assert all(type(layer.mlp._module).__name__ == "FlexOlmoSparseMoeBlock" for layer in model.layers)
