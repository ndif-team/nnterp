"""DeepSeek-V2, end to end: multi-head latent attention."""

from suite import FamilySuite, LLAMA_ROWS

from nnterp.families import deepseek_v2


class TestDeepseekV2(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-DeepseekV2ForCausalLM"
    FAMILY = deepseek_v2
    NATIVE = LLAMA_ROWS
    KV_HEADS_EXPANDED = True  # latent attention projects keys and values for every head

    def test_latent_attention_widths(self, model):
        assert model.qk_head_dim == model.config.qk_nope_head_dim + model.config.qk_rope_head_dim
        assert model.head_dim == model.config.v_head_dim
