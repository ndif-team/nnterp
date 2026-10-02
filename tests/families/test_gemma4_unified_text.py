"""Gemma 4 unified, text, end to end: Gemma-4's sandwich and layer_scalar, KV sharing, no per-layer embeddings or experts."""

from test_gemma4_text import Gemma4Suite

from nnter.families import gemma4_unified_text


class TestGemma4Unified(Gemma4Suite):
    REPO = "hf-tiny-v2/tiny-random-Gemma4UnifiedForCausalLM"
    FAMILY = gemma4_unified_text

    def test_no_per_layer_output(self, model):
        assert "per_layer_output" not in model.support()
        assert not hasattr(gemma4_unified_text.Layer, "per_layer_output")
