"""GPT-2, end to end."""

from suite import FamilySuite, rows

from nnterp.families import gpt2


class TestGPT2(FamilySuite):
    REPO = "hf-internal-testing/tiny-random-gpt2"
    FAMILY = gpt2
    NATIVE = rows("transformer", "h", "wte", "ln_f", attn="attn", ln1="ln_1", ln2="ln_2")
    REFUSES_IN_PLACE_QKV = True  # q/k/v are split views of one c_attn tensor

    def test_reorder_and_upcast_makes_the_interface_unavailable(self, model):
        """That config flag takes GPT-2's own upcast path, where nothing on the interface runs."""
        config = model.layers[0].self_attn._module.config
        config.reorder_and_upcast_attn = True
        try:
            support = model.support()
            assert all("reorder_and_upcast_attn" in support[f"self_attn.{name}"][0] for name in ("attention_probabilities", "attention_queries"))
        finally:
            config.reorder_and_upcast_attn = False
