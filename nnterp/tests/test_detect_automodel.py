"""Tests for detect_automodel auto-detection logic."""

import inspect

from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
)

from nnterp import detect_automodel
from nnterp.standardized_transformer import StandardizedTransformer


def test_detect_automodel_causal_lm():
    """Pure text CausalLM model should get AutoModelForCausalLM."""
    assert detect_automodel("Qwen/Qwen3-0.6B") is AutoModelForCausalLM


def test_detect_automodel_vlm():
    """VLM model should get AutoModelForImageTextToText, not CausalLM."""
    assert detect_automodel("Qwen/Qwen2-VL-2B-Instruct") is AutoModelForImageTextToText


def test_detect_automodel_seq2seq():
    """Seq2Seq model should get AutoModelForSeq2SeqLM."""
    assert detect_automodel("google-t5/t5-small") is AutoModelForSeq2SeqLM


def test_detect_automodel_explicit_override():
    """StandardizedTransformer.__init__ accepts automodel param with default None."""
    sig = inspect.signature(StandardizedTransformer.__init__)
    assert "automodel" in sig.parameters
    assert sig.parameters["automodel"].default is None
