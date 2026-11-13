"""Granite SWA, end to end: Granite's scaled adds, sliding-window blocks, a sink applied after the softmax."""

import test_granite
import torch
from suite import FamilySuite, LLAMA_ROWS, PROMPT, built

from nnterp.families import granite_swa

REPO = "hf-tiny-v2/tiny-random-GraniteSWAForCausalLM"


def sink_scaled_head_outputs(model, layer):
    """The head outputs recomputed from the served pattern and values, scaled by the sink's share, as the eager forward does."""
    attn = layer.self_attn
    with model.trace(PROMPT):
        scores = attn.attention_scores.save()
    with model.trace(PROMPT):
        values = attn.attention_values.save()
    with model.trace(PROMPT):
        probs = attn.attention_probabilities.save()
    with model.trace(PROMPT):
        heads = attn.attention_head_outputs.save()
    module = attn._module
    values = values.repeat_interleave(module.num_key_value_groups, dim=1)
    scale = (torch.logsumexp(scores, dim=-1) - module.sinks.view(1, -1, 1)).float().sigmoid()
    expected = (probs @ values) * scale.unsqueeze(-1).to(probs.dtype)
    return expected.transpose(1, 2), heads, probs


def written_scores_and_head_outputs(model, layer):
    """Scores with key 0 masked for every query but the first, written in place, and the head outputs the block then
    returns, beside what attending with the written scores gives: their softmax mixing the values, times the share
    ``sigmoid(logsumexp(written) - sink)`` the written scores imply."""
    attn = layer.self_attn
    with model.trace(PROMPT):
        values = attn.attention_values.save()
        scores = attn.attention_scores
        scores[:, :, 1:, 0] = torch.finfo(scores.dtype).min
        written = scores.clone().save()
        heads = attn.attention_head_outputs.save()
    module = attn._module
    values = values.repeat_interleave(module.num_key_value_groups, dim=1)
    share = (torch.logsumexp(written, dim=-1) - module.sinks.view(1, -1, 1)).float().sigmoid()
    expected = (written.float().softmax(-1).to(values.dtype) @ values) * share.unsqueeze(-1).to(values.dtype)
    return expected.transpose(1, 2), heads


class TestGraniteSWA(FamilySuite):
    REPO = REPO
    FAMILY = granite_swa
    NATIVE = LLAMA_ROWS

    def test_block_kinds(self, model):
        windows = [layer.self_attn._module.sliding_window for layer in model.layers]
        assert windows == [model.config.sliding_window if kind == "sliding_attention" else None for kind in model.config.layer_types]

    def test_sink_scales_the_head_outputs_not_the_pattern(self, model):
        for layer in model.layers:
            expected, heads, probs = sink_scaled_head_outputs(model, layer)
            torch.testing.assert_close(probs.sum(-1), torch.ones_like(probs.sum(-1)))
            torch.testing.assert_close(heads, expected)

    def test_written_scores_reach_the_share(self, model):
        """The share is computed from the scores, so a write to them moves it with the pattern."""
        for layer in model.layers:
            expected, heads = written_scores_and_head_outputs(model, layer)
            torch.testing.assert_close(heads, expected)


class TestGraniteSWAScaled(test_granite.TestGraniteScaled):
    """The same weights with every multiplier away from 1.0 (Granite's rewrite)."""

    REPO = built(test_granite._scaled_checkpoint, REPO)
    FAMILY = granite_swa
    NATIVE = LLAMA_ROWS
