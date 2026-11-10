"""GraniteMoE SWA, end to end: Granite SWA's attention in GraniteMoE's scaled block."""

import test_granite
import torch
from suite import FamilySuite, LLAMA_ROWS
from test_granite_swa import sink_scaled_head_outputs, written_scores_and_head_outputs
from test_granitemoeshared import written_mlp_input_reaches

from nnterp import StandardizedTransformer
from nnterp.families import granitemoe_swa

REPO = "hf-tiny-v2/tiny-random-GraniteMoeSWAForCausalLM"
NATIVE = {**LLAMA_ROWS, "layers.0.mlp": "model.layers.0.block_sparse_moe"}


class TestGraniteMoeSWA(FamilySuite):
    REPO = REPO
    FAMILY = granitemoe_swa
    NATIVE = NATIVE

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


class TestGraniteMoeSWAScaled(test_granite.TestGraniteScaled):
    """The same weights with every multiplier away from 1.0 (Granite's rewrite)."""

    REPO = test_granite._scaled_checkpoint(REPO)
    FAMILY = granitemoe_swa
    NATIVE = NATIVE


def test_mlp_input_reaches_the_shared_expert():
    """With ``shared_intermediate_size`` set (granite-swash-3b-a600m's layout; the pinned tiny has 0) the block calls a
    ``shared_mlp`` beside the mixture with the same normed tensor, and a write to ``mlp_input`` reaches both. The
    shared expert's weights are not in the tiny checkpoint, so they load at their random initialization."""
    model = StandardizedTransformer(test_granite._scaled_checkpoint(REPO, shared_intermediate_size=16), dispatch=True, attn_implementation="eager")
    layer = model.layers[0]
    assert layer._module.shared_mlp is not None
    written_mlp_input_reaches(model, layer, layer.mlp, layer.shared_mlp)

