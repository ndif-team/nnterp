"""GraniteMoE-Hybrid, end to end: Mamba-2 or attention per block, Granite's scaled adds, a feed-forward of two experts."""

import test_granite
import torch
from ssd import StateSpaceChecks
from suite import LINEAR, LLAMA_ROWS, PROMPT, VALUES, FamilySuite

from nnter.families import granitemoehybrid

REPO = "hf-tiny-v2/tiny-random-GraniteMoeHybridForCausalLM"
SSD_BLOCKS = (0, 2)       # ``layer_types`` = linear, full, linear, full
ATTENTION_BLOCKS = (1, 3)


def scaled_mixture(self, model, host):
    """The block's sum of the mixture and the shared expert: ``mlp_output`` over the multiplier."""
    with model.trace(PROMPT):
        out = host.mlp_output.save()
    return out / model.config.residual_multiplier


class HybridSuite(StateSpaceChecks, FamilySuite):
    FAMILY = granitemoehybrid
    mixture_output = scaled_mixture
    NATIVE = {
        **{k: v for k, v in LLAMA_ROWS.items() if k != "layers.0.self_attn"},
        "layers.0.mlp": "model.layers.0.shared_mlp",
        "layers.0.linear_attn": "model.layers.0.mamba",
        "layers.1.self_attn": "model.layers.1.self_attn",
    }
    EXPECTED_UNAVAILABLE = {
        **{name: "no self_attn module" for name in VALUES if name.startswith("self_attn.")},
        **{f"linear_attn.{name}": "no linear_attn module" for name in LINEAR if name not in ("state", "states")},
        "linear_attn.state": "",   # missing off the Mamba-2 blocks, and unavailable on them
        "linear_attn.states": "",
        "linear_attn.set_state_after": "",
    }

    def test_every_layer_is_renamed(self, model):
        for i, layer in enumerate(model.layers):
            assert hasattr(layer, "linear_attn") == (i in SSD_BLOCKS)
            assert (getattr(layer, "self_attn", None) is not None) == (i in ATTENTION_BLOCKS)
            assert layer.mlp is layer.shared_mlp

    def test_mlp_output_is_the_whole_feed_forward(self, model):
        scale = model.config.residual_multiplier
        for layer in model.layers:
            routed = None  # bound outside: a name bound inside the block does not survive it
            with model.trace(PROMPT):
                if layer._module.has_experts:  # the routed experts run before the shared one
                    routed = layer.block_sparse_moe.output.save()
                shared = layer.mlp.output.save()
                out = layer.mlp.mlp_output.save()
            torch.testing.assert_close(out, (shared if routed is None else routed + shared) * scale)


class TestGraniteMoeHybrid(HybridSuite):
    REPO = REPO


class TestGraniteMoeHybridScaled(HybridSuite):
    """Granite's multipliers away from 1.0: the identity holds only on the scaled terms, on both block kinds."""

    REPO = test_granite._scaled_checkpoint(REPO)

    def test_mixer_contributions_are_scaled(self, model):
        with model.trace(PROMPT):
            ssd_raw = model.layers[0].linear_attn.output.save()
            ssd = model.layers[0].linear_attn.attention_output.save()
        with model.trace(PROMPT):
            attn_raw = model.layers[1].self_attn.output[0].save()
            attn = model.layers[1].self_attn.attention_output.save()
        torch.testing.assert_close(ssd, ssd_raw * 0.22)
        torch.testing.assert_close(attn, attn_raw * 0.22)

    def test_in_place_edits_reach_the_model(self, model):
        with model.trace(PROMPT):
            layer = model.layers[0]
            x = layer.input.save()
            ssd = layer.linear_attn.attention_output
            ssd[:] = 0
            layer.mlp.mlp_output[:] = 0
            out = layer.layer_output.save()
        torch.testing.assert_close(out, x)

    def test_logits_are_the_head_over_logits_scaling(self, model):
        with model.trace(PROMPT):
            resid = model.layers[-1].layer_output.save()
            raw = model.lm_head.output.save()
            logits = model.logits.save()
        torch.testing.assert_close(logits, raw / 4.0)
        torch.testing.assert_close(model.project_on_vocab(resid), logits)


class TestGraniteMoeHybridDense(HybridSuite):
    """``num_local_experts`` 0, as Granite-4.0-H-Micro: the shared expert is the whole feed-forward."""

    REPO = test_granite._scaled_checkpoint(REPO, num_local_experts=0)

    def test_no_routed_experts(self, model):
        assert all(layer._module.block_sparse_moe is None for layer in model.layers)
