"""Gemma 3 text, end to end: a sandwich block, as ``Gemma3ForCausalLM`` and inside a ``Gemma3ForConditionalGeneration``."""

import torch
from suite import FamilySuite, LLAMA_ROWS, PROMPT, rows
from vision_suite import IMAGE, VisionSuite, image_prompt

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


class TestGemma3ImageTextToText(TestGemma3Wrapper):
    """The wrapper loaded with its processor: the whole suite on the text side, the image values listed."""

    LOAD_KWARGS = {"task": "image-text-to-text"}


class TestGemma3Vision(VisionSuite):
    """Gemma 3's SigLIP tower and pooling projector, and the tower's image values.

    ``yujiepan/gemma-3-tiny-random``: the trl tiny wrapper's projector outputs
    exact zeros, so no edit upstream of it would show there.
    """

    REPO = "yujiepan/gemma-3-tiny-random"
    FAMILY = gemma3_text
    TEXT_REPO = TestGemma3.REPO
    VISION_NATIVE = {
        "vision": "model.vision_tower",
        "vision.layers": "model.vision_tower.encoder.layers",
        "vision.patch_embed": "model.vision_tower.embeddings.patch_embedding",
        "vision.norm": "model.vision_tower.post_layernorm",
        "vision.layers.0.self_attn": "model.vision_tower.encoder.layers.0.self_attn",
        "vision.layers.0.mlp": "model.vision_tower.encoder.layers.0.mlp",
        "vision.layers.0.input_layernorm": "model.vision_tower.encoder.layers.0.layer_norm1",
        "vision.layers.0.post_attention_layernorm": "model.vision_tower.encoder.layers.0.layer_norm2",
        "projector": "model.multi_modal_projector",
    }

    def test_the_projector_pools_the_tower_output(self, model):
        with model.trace(image_prompt(model), images=[IMAGE]):
            out = model.vision.tower_output.save()
            fed = model.projector.input.save()
            features = model.vision.image_features.save()
        assert torch.equal(fed, out)
        assert features.shape[0] == model.config.mm_tokens_per_image < out.shape[1]
