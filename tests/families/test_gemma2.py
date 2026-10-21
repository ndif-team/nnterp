"""Gemma 2, end to end: a sandwich block with logit softcapping; and PaliGemma 2, the SigLIP wrapper around it."""

import os
import tempfile

import pytest
import torch
from suite import FamilySuite, LLAMA_ROWS, PROMPT
from vision_suite import IMAGE, VisionSuite, WrapperSuite, image_prompt, siglip_rows

from nnterp import StandardizedTransformer
from nnterp.families import gemma2


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


TEXT_REPO = TestGemma2.REPO
#: PaliGemma 1's tiny wrapper: its SigLIP config and its processor (the PaliGemma tokenizer and a 20-pixel image processor).
PALIGEMMA_REPO = "hf-tiny-v2/tiny-random-PaliGemmaForConditionalGeneration"


def _paligemma2_checkpoint():
    """A tiny PaliGemma 2 (``PaliGemmaForConditionalGeneration`` over the tiny Gemma 2) with its processor, random
    weights; written once per text snapshot.

    No tiny PaliGemma 2 is published. The text config is the tiny Gemma 2's with PaliGemma's vocabulary
    (257216, as on every PaliGemma 2), the tower is PaliGemma 1's tiny SigLIP (20 pixels in 5-pixel patches: 16
    image tokens) and the projector maps it onto the text width; the processor is PaliGemma 1's tiny one with
    ``image_seq_length`` set to the 16 tokens, and the config's image token is the processor's.
    """
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, PaliGemmaConfig, PaliGemmaProcessor

    cache = os.path.expanduser(f"~/.cache/huggingface/hub/models--{TEXT_REPO.replace('/', '--')}")
    snapshot = open(os.path.join(cache, "refs", "main")).read().strip()
    built = os.path.join(tempfile.gettempdir(), f"nnterp-paligemma2-{snapshot}")
    if os.path.exists(os.path.join(built, "processor_config.json")):
        return built
    text = AutoConfig.from_pretrained(os.path.join(cache, "snapshots", snapshot))
    text.vocab_size = 257216
    processor = AutoProcessor.from_pretrained(PALIGEMMA_REPO)
    vision = AutoConfig.from_pretrained(PALIGEMMA_REPO).vision_config
    processor.image_processor.image_seq_length = (vision.image_size // vision.patch_size) ** 2
    config = PaliGemmaConfig(
        text_config=text.to_dict(), vision_config=vision.to_dict(), projection_dim=text.hidden_size,
        image_token_index=processor.image_token_id,
    )
    torch.manual_seed(0)
    partial = tempfile.mkdtemp(prefix="nnterp-paligemma2-")
    AutoModelForImageTextToText.from_config(config).save_pretrained(partial)
    PaliGemmaProcessor(image_processor=processor.image_processor, tokenizer=processor.tokenizer).save_pretrained(partial)
    try:
        os.replace(partial, built)
    except OSError:  # another process built it first
        pass
    return built


PALIGEMMA2 = _paligemma2_checkpoint()


class TestPaliGemma2Wrapper(WrapperSuite):
    """PaliGemma 2: the Gemma 2 text stack at ``model.language_model``. Its processor refuses a prompt without an image,
    so the text-only trace takes the tokenizer's encoding."""

    FAMILY = gemma2

    @pytest.fixture(scope="class")
    def model(self):
        return StandardizedTransformer(PALIGEMMA2, task="image-text-to-text", dispatch=True, attn_implementation="eager")

    def text_input(self, model):
        return dict(model.tokenizer(PROMPT, return_tensors="pt"))

    def test_the_wrapper_does_not_softcap_and_neither_does_the_lens(self, model):
        """``text_config.final_logit_softcapping`` is set, but ``PaliGemmaForConditionalGeneration`` never applies it."""
        assert model.config.get_text_config().final_logit_softcapping == 30.0
        with model.trace(self.text_input(model)):
            last = model.layers[-1].layer_output.save()
            raw = model.lm_head.output.save()
            logits = model.logits.save()
        assert torch.equal(raw, logits)
        torch.testing.assert_close(model.project_on_vocab(last), logits)

    def test_the_prefix_attends_in_both_directions(self, model):
        """The processor marks the image and the prompt as the prefix: every block, sliding or full, attends over all of it."""
        with model.trace(image_prompt(model), images=[IMAGE]):
            sliding = model.layers[0].self_attn.attention_probabilities.save()
            full = model.layers[1].self_attn.attention_probabilities.save()
        assert model.config.get_text_config().layer_types[:2] == ["sliding_attention", "full_attention"]
        assert (sliding > 0).all() and (full > 0).all()


class TestPaliGemma2Vision(VisionSuite):
    """PaliGemma 2's SigLIP tower and linear projector, and the tower's image values, on `_paligemma2_checkpoint`."""

    REPO = PALIGEMMA2
    FAMILY = gemma2
    TEXT_REPO = TEXT_REPO
    VISION_NATIVE = siglip_rows()

    def text_input(self, model):
        return dict(model.tokenizer(PROMPT, return_tensors="pt"))


def test_the_paligemma2_3b_names_bind_on_meta():
    """``google/paligemma2-3b-pt-224``'s own config and processor (no weights cached), built on meta: the text names, the
    tower and the projector bind, and the sizes are the Gemma 2 2B's and the SigLIP's."""
    model = StandardizedTransformer("google/paligemma2-3b-pt-224", task="image-text-to-text")
    assert model.family is gemma2 and type(model._module).__name__ == "PaliGemmaForConditionalGeneration"
    assert model.layers is model.get("model.language_model.layers") and model.vision is model.get("model.vision_tower")
    assert model.projector is model.get("model.multi_modal_projector")
    assert (model.num_layers, model.hidden_size) == (26, 2304)
    assert (model.vision.num_layers, model.vision.vision_hidden_size, model.vision.image_size, model.vision.patch_size) == (27, 1152, 224, 14)
    assert {"vision.image_token_mask", "vision.image_features"} <= set(model.support())
