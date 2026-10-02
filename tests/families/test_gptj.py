"""GPT-J, end to end: tuple block output, attention arithmetic in ``_attn``."""

from suite import FamilySuite, rows, PROMPT

from nnter.families import gptj


class TestGPTJ(FamilySuite):
    REPO = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    FAMILY = gptj
    NATIVE = rows("transformer", "h", "wte", "ln_f", attn="attn", ln1="ln_1", ln2=None)
    MLP_NORM = "input_layernorm"             # parallel: ln_1 feeds both sublayers

    def test_block_returns_a_tuple(self, model):
        with model.trace(PROMPT):
            raw = model.layers[0].output.save()
        assert isinstance(raw, tuple)
