"""MiMo-V2-Flash, end to end: split query/value widths, sinks on the sliding-window blocks."""

import torch
from suite import FamilySuite, LLAMA_ROWS, PROMPT

from nnterp.families import mimo_v2_flash


def with_sink(scores, sinks):
    """The sliding blocks' softmax: the sink as one extra key column, the row max subtracted, the column dropped."""
    batch, heads, q, _ = scores.shape
    column = sinks.to(scores.dtype).view(1, heads, 1, 1).expand(batch, heads, q, 1)
    combined = torch.cat([scores, column], dim=-1)
    combined = combined - combined.max(dim=-1, keepdim=True).values
    return combined.softmax(-1)[..., :-1]


class TestMiMoV2Flash(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-MiMoV2FlashForCausalLM"
    FAMILY = mimo_v2_flash
    NATIVE = LLAMA_ROWS  # block 0 is full attention: no sink there, so the suite's pattern checks hold as on Llama

    def test_split_widths(self, model):
        assert model.qk_head_dim == model.config.head_dim != model.head_dim == model.config.v_head_dim
        with model.trace(PROMPT):
            values = model.layers[0].self_attn.attention_values.save()
        assert values.shape[-1] == model.head_dim

    def test_sliding_blocks_have_the_sink_and_twice_the_kv_heads(self, model):
        kinds = model.config.layer_types
        for layer, kind in zip(model.layers, kinds):
            attn = layer.self_attn._module
            assert (attn.sinks is not None) == (kind == "sliding_attention")
        sliding = model.layers[kinds.index("sliding_attention")].self_attn
        with model.trace(PROMPT):
            scores = sliding.attention_scores.save()
        with model.trace(PROMPT):
            keys = sliding.attention_keys.save()
        with model.trace(PROMPT):
            probs = sliding.attention_probabilities.save()
        assert keys.shape[1] == sliding.num_kv_heads == 2 * model.num_kv_heads
        sums = probs.sum(-1)
        assert (sums < 1).all() and (sums > 0).all()
        torch.testing.assert_close(with_sink(scores, sliding._module.sinks), probs)

    def test_dense_then_mixture(self, model):
        kinds = [type(layer.mlp._module).__name__ for layer in model.layers]
        assert kinds == [{"dense": "MiMoV2FlashMLP", "sparse": "MiMoV2FlashMoE"}[t] for t in model.config.mlp_layer_types]
