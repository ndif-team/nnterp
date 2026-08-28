"""Tests for detect_automodel and vision LLM support."""

from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
)

from nnterp import detect_automodel


def test_detect_automodel_causal_lm():
    """Pure text CausalLM model should get AutoModelForCausalLM."""
    assert detect_automodel("Qwen/Qwen3-0.6B") is AutoModelForCausalLM


def test_detect_automodel_vlm():
    """VLM model should get AutoModelForImageTextToText, not CausalLM."""
    assert detect_automodel("Qwen/Qwen2-VL-2B-Instruct") is AutoModelForImageTextToText


def test_detect_automodel_seq2seq():
    """Seq2Seq model should get AutoModelForSeq2SeqLM."""
    assert detect_automodel("google-t5/t5-small") is AutoModelForSeq2SeqLM


import pytest
from transformers import MllamaForCausalLM

from nnterp import StandardizedTransformer, load_model

# Mllama registers MllamaForCausalLM next to MllamaForConditionalGeneration
MLLAMA_TINY = "yujiepan/llama-3.2-vision-tiny-random"


def test_detect_automodel_text_only_selects_text_tower():
    assert detect_automodel(MLLAMA_TINY) is AutoModelForImageTextToText
    assert detect_automodel(MLLAMA_TINY, text_only=True) is AutoModelForCausalLM


def test_detect_automodel_text_only_without_separate_text_class():
    """gemma3 maps to Gemma3ForConditionalGeneration in both auto tables."""
    assert (
        detect_automodel("yujiepan/gemma-3-tiny-random", text_only=True)
        is AutoModelForImageTextToText
    )


def test_detect_automodel_text_only_unbuildable_text_tower_raises():
    """Llama4ForCausalLM cannot be built from the composite Llama4Config in
    transformers 4.56. If a newer transformers fixes that, this should start
    returning a CausalLM subclass instead and the test needs updating."""
    with pytest.raises(ValueError, match="text_only=True"):
        detect_automodel("yujiepan/llama-4-tiny-random", text_only=True)


def test_text_only_loads_text_tower():
    # Mllama's text tower has cross-attention layers (heterogeneous), so skip renaming checks
    model = load_model(MLLAMA_TINY, text_only=True, check_renaming=False)
    assert isinstance(model, StandardizedTransformer)
    assert isinstance(model._model, MllamaForCausalLM)
    assert not hasattr(model._model, "vision_model")
    with model.trace("Hello, world!"):
        logits = model.logits.save()
    assert logits.shape[-1] == model.vocab_size
