"""GPT-NeoX (Pythia), end to end."""

import pytest
from suite import FamilySuite, built, patched_checkpoint, rows

from nnterp.families import gpt_neox


class TestGPTNeoX(FamilySuite):
    REPO = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"
    FAMILY = gpt_neox
    NATIVE = rows("gpt_neox", "layers", "embed_in", "final_layer_norm", attn="attention")
    MLP_NORM = "post_attention_layernorm"   # parallel: normalizes the block input, not a mid-stream
    EXPECTED_UNAVAILABLE = {"layer_mid": "a parallel block"}   # use_parallel_residual on the tiny checkpoint

    def test_block_is_parallel(self, model):
        """Pythia's block is ``x + attn(ln1(x)) + mlp(ln2(x))``; the contribution identity holds regardless."""
        assert model.config.use_parallel_residual


class TestGPTNeoXSequential(TestGPTNeoX):
    """``use_parallel_residual`` false: the sequential block ``x1 = x + attn(ln1(x)); x1 + mlp(ln2(x1))``, where
    ``layer_mid`` is served. No released checkpoint takes it; the tiny checkpoint's weights."""

    REPO = built(patched_checkpoint, TestGPTNeoX.REPO, use_parallel_residual=False)
    MLP_NORM = "post_attention_layernorm"
    EXPECTED_UNAVAILABLE = {}

    def test_block_is_parallel(self, model):
        pytest.skip("a sequential block")

    def test_block_is_sequential(self, model):
        assert not model.config.use_parallel_residual and model.support(layer=0)["layer_mid"] is None
