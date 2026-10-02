"""GPT-OSS, end to end: attention sinks, a mixture of experts returning router scores."""

import torch
from suite import FamilySuite, LLAMA_ROWS, PROMPT

from nnter.families import gpt_oss


class TestGptOss(FamilySuite):
    REPO = "yujiepan/gpt-oss-tiny-random"
    FAMILY = gpt_oss
    NATIVE = LLAMA_ROWS
    ATTENTION_SINK = True

    def pattern_from_scores(self, model, scores):
        """The sink joins the softmax as one extra key column and is dropped afterwards.

        GPT-OSS takes this softmax in the scores' own dtype after subtracting
        the row max, not in float32, so the reference does the same.
        """  # noqa
        sinks = model.layers[0].self_attn._module.sinks.to(scores.dtype)
        batch, heads, q, _ = scores.shape
        column = sinks.view(1, heads, 1, 1).expand(batch, heads, q, 1)
        combined = torch.cat([scores, column], dim=-1)
        combined = combined - combined.max(dim=-1, keepdim=True).values
        return combined.softmax(-1)[..., :-1]

    def test_mlp_returns_router_scores_beside_the_hidden_states(self, model):
        with model.trace(PROMPT):
            raw = model.layers[0].mlp.output.save()
            std = model.layers[0].mlp.mlp_output.save()
        assert isinstance(raw, tuple) and len(raw) == 2 and torch.equal(std, raw[0])
