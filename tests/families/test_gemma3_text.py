"""Gemma 3 text, end to end: a sandwich block, as ``Gemma3ForCausalLM`` and inside a ``Gemma3ForConditionalGeneration``."""

import torch
from suite import FamilySuite, LLAMA_ROWS, PROMPT, rows

from nnterp.families import gemma3_text


class TestGemma3(FamilySuite):
    REPO = "hf-internal-testing/tiny-random-Gemma3ForCausalLM"
    FAMILY = gemma3_text
    NATIVE = LLAMA_ROWS
    MLP_NORM = "pre_feedforward_layernorm"   # sandwich: post_attention_layernorm follows the attention

    def test_contributions_are_the_post_norms(self, model):
        with model.trace(PROMPT):
            attn = model.layers[0].self_attn.attention_output.save()
            post_attn = model.layers[0].post_attention_layernorm.output.save()
        assert torch.equal(attn, post_attn)


class TestGemma3Wrapper(TestGemma3):
    """A ``gemma3`` checkpoint: the text stack at ``model.language_model``, the sizes in ``text_config``."""

    REPO = "trl-internal-testing/tiny-Gemma3ForConditionalGeneration"
    NATIVE = rows("model.language_model", "layers", "embed_tokens", "norm")

    def test_the_text_stack_is_under_language_model(self, model):
        assert type(model._module).__name__ == "Gemma3ForConditionalGeneration"
        assert model.config.model_type == "gemma3"
        assert model.layers is model.get("model.language_model.layers")
        assert model.embed_tokens is model.get("model.language_model.embed_tokens")
        assert model.norm is model.get("model.language_model.norm")
        text = model.config.text_config
        assert (model.hidden_size, model.num_heads, model.num_kv_heads, model.head_dim) == (
            text.hidden_size, text.num_attention_heads, text.num_key_value_heads, text.head_dim)


class TestGemma3WrapperWithACap(TestGemma3Wrapper):
    """A wrapper whose text config sets a softcap: the model does not apply it, so the lens does not either."""

    def test_the_lens_matches_the_uncapped_logits(self, model):
        model.config.text_config.final_logit_softcapping = 30.0
        try:
            with model.trace(PROMPT):
                out = model.layers[-1].layer_output.save()
                logits = model.logits.save()
            torch.testing.assert_close(model.project_on_vocab(out), logits)
        finally:
            model.config.text_config.final_logit_softcapping = None
