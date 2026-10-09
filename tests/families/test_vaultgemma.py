"""VaultGemma, end to end: Gemma's names with pre-norms only and logit softcapping (the pinned tiny sets the caps; google/vaultgemma-1b sets them to null)."""

import torch
from suite import FamilySuite, LLAMA_ROWS, PROMPT

from nnterp.families import vaultgemma


class TestVaultGemma(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-VaultGemmaForCausalLM"
    FAMILY = vaultgemma
    NATIVE = {**LLAMA_ROWS, "layers.0.post_attention_layernorm": "model.layers.0.pre_feedforward_layernorm"}

    def test_contributions_are_the_module_outputs(self, model):
        with model.trace(PROMPT):
            attn_raw = model.layers[0].self_attn.output[0].save()
            attn = model.layers[0].self_attn.attention_output.save()
            mlp_raw = model.layers[0].mlp.output.save()
            mlp = model.layers[0].mlp.mlp_output.save()
        assert torch.equal(attn, attn_raw) and torch.equal(mlp, mlp_raw)

    def test_logits_are_softcapped(self, model):
        assert model.config.final_logit_softcapping
        with model.trace(PROMPT):
            raw = model.lm_head.output.save()
            logits = model.logits.save()
        assert not torch.equal(raw, logits)
