"""OLMo 2, end to end: post-norms only."""

import torch
from suite import FamilySuite, rows, PROMPT

from nnter.families import olmo2


class TestOlmo2(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-Olmo2ForCausalLM"
    FAMILY = olmo2
    NATIVE = rows("model", "layers", "embed_tokens", "norm", ln1=None)
    ATTENTION_NORM = None                    # post-norms only: the block input enters the attention
    MLP_NORM = None                          # ... and the MLP takes the residual stream after the attention add

    def test_contributions_are_the_post_norms(self, model):
        with model.trace(PROMPT):
            attn = model.layers[0].self_attn.attention_output.save()
            post_attn = model.layers[0].post_attention_layernorm.output.save()
        assert torch.equal(attn, post_attn)
