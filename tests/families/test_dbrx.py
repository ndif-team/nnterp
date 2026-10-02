"""DBRX, end to end: the attention wrapped in ``norm_attn_norm``, a mixture-of-experts FFN."""

import torch
from suite import FamilySuite, PROMPT, rows

from nnter.families import dbrx


class TestDbrx(FamilySuite):
    REPO = "yujiepan/dbrx-tiny256-random"
    FAMILY = dbrx
    NATIVE = rows("transformer", "blocks", "wte", "norm_f", attn="norm_attn_norm.attn", mlp="ffn", ln1="norm_attn_norm.norm_1", ln2="norm_attn_norm.norm_2")
    MOE_UNAVAILABLE = {"expert_outputs": "loop over the experts"}
    ATTENTION_NORM = "norm_attn_norm.norm_1"
    MLP_NORM = "norm_attn_norm.norm_2"
    #: The tiny checkpoint is fp16 with weights of std 0.02: its attention scores round to a
    #: uniform pattern, which makes the queries inert and every layer's pattern identical.
    LOAD_KWARGS = {"dtype": torch.float32}

    def test_sizes_match_the_model(self, model):
        """DBRX keeps the FFN width in ``ffn_config.ffn_hidden_size``; the rest as the suite checks."""
        model.config.intermediate_size = model.config.ffn_config.ffn_hidden_size
        super().test_sizes_match_the_model(model)

    def test_residual_is_added_outside_the_attention_module(self, model):
        with model.trace(PROMPT):
            x = model.layers[0].input.save()
            attn = model.layers[0].self_attn.attention_output.save()
            after = model.layers[0].norm_attn_norm.output.save()
        torch.testing.assert_close(after[0], x + attn)
