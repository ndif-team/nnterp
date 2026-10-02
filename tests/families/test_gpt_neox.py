"""GPT-NeoX (Pythia), end to end."""

from suite import FamilySuite, rows

from nnter.families import gpt_neox


class TestGPTNeoX(FamilySuite):
    REPO = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"
    FAMILY = gpt_neox
    NATIVE = rows("gpt_neox", "layers", "embed_in", "final_layer_norm", attn="attention")
    MLP_NORM = "post_attention_layernorm"   # parallel: normalizes the block input, not a mid-stream

    def test_block_is_parallel(self, model):
        """Pythia's block is ``x + attn(ln1(x)) + mlp(ln2(x))``; the contribution identity holds regardless."""
        assert model.config.use_parallel_residual
