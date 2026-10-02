"""Qwen3, end to end."""

from suite import FamilySuite, LLAMA_ROWS

from nnter.families import qwen3


class TestQwen3(FamilySuite):
    REPO = "trl-internal-testing/tiny-Qwen3ForCausalLM"
    FAMILY = qwen3
    NATIVE = LLAMA_ROWS

    def test_head_dim_is_the_configs_not_hidden_over_heads(self, model):
        assert model.head_dim == model.config.head_dim != model.hidden_size // model.num_heads
