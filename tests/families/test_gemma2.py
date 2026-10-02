"""Gemma 2, end to end: a sandwich block with logit softcapping."""

import torch
from suite import FamilySuite, LLAMA_ROWS, PROMPT

from nnter.families import gemma2


class TestGemma2(FamilySuite):
    REPO = "trl-internal-testing/tiny-Gemma2ForCausalLM"
    FAMILY = gemma2
    NATIVE = LLAMA_ROWS
    MLP_NORM = "pre_feedforward_layernorm"   # sandwich: post_attention_layernorm follows the attention

    def test_contributions_are_the_post_norms(self, model):
        with model.trace(PROMPT):
            attn = model.layers[0].self_attn.attention_output.save()
            post_attn = model.layers[0].post_attention_layernorm.output.save()
            mlp = model.layers[0].mlp.mlp_output.save()
            post_ff = model.layers[0].post_feedforward_layernorm.output.save()
        assert torch.equal(attn, post_attn) and torch.equal(mlp, post_ff)

    def test_logits_are_softcapped(self, model):
        assert model.config.final_logit_softcapping
        with model.trace(PROMPT):
            raw = model.lm_head.output.save()
            logits = model.logits.save()
        assert not torch.equal(raw, logits)
