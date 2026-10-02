"""Youtu-LLM, end to end: multi-head latent attention on dense blocks."""

from suite import FamilySuite, LLAMA_ROWS

from nnterp.families import youtu


class TestYoutu(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-YoutuForCausalLM"
    FAMILY = youtu
    NATIVE = LLAMA_ROWS
    KV_HEADS_EXPANDED = True  # latent attention projects keys and values for every head

    def test_latent_attention_widths(self, model):
        assert model.qk_head_dim == model.config.qk_nope_head_dim + model.config.qk_rope_head_dim
        assert model.head_dim == model.config.v_head_dim
        assert model.config.head_dim == model.config.qk_rope_head_dim  # the config's alias is the rotary part
