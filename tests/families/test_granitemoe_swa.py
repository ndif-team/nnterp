"""GraniteMoE SWA, end to end: Granite SWA's attention in GraniteMoE's scaled block."""

import test_granite
import torch
from suite import FamilySuite, LLAMA_ROWS
from test_granite_swa import sink_scaled_head_outputs

from nnter.families import granitemoe_swa

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


class TestGraniteMoeSWAScaled(test_granite.TestGraniteScaled):
    """The same weights with every multiplier away from 1.0 (Granite's rewrite)."""

    REPO = test_granite._scaled_checkpoint(REPO)
    FAMILY = granitemoe_swa
    NATIVE = NATIVE
