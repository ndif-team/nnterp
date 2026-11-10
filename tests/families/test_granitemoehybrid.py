"""GraniteMoE-Hybrid, end to end: Mamba-2 or attention per block, Granite's scaled adds, a feed-forward of two experts."""

import json
import os

import pytest
import test_granite
import torch
from ssd import StateSpaceChecks
from suite import LINEAR, LLAMA_ROWS, NO_ROTARY, PROMPT, VALUES, FamilySuite, blocks, cached_snapshot, patched_checkpoint, built
from test_granitemoeshared import written_mlp_input_reaches

from nnterp.families import granitemoehybrid

REPO = "hf-tiny-v2/tiny-random-GraniteMoeHybridForCausalLM"   # ``layer_types`` = linear, full, linear, full


def scaled_mixture(self, model, host):
    """The block's sum of the mixture and the shared expert: ``mlp_output`` over the multiplier."""
    with model.trace(PROMPT):
        out = host.mlp_output.save()
    return out / model.config.residual_multiplier


class GraniteMoeHybridSuite(FamilySuite):
    """What every layout shares: the blocks of each kind read from ``layer_types``, the feed-forward under ``shared_mlp``."""

    FAMILY = granitemoehybrid
    mixture_output = scaled_mixture
    NATIVE = {**LLAMA_ROWS, "layers.0.mlp": "model.layers.0.shared_mlp"}

    def native(self, model):
        """`NATIVE` with the first block of each mixer kind where the config puts it."""
        rows = {k: v for k, v in self.NATIVE.items() if k != "layers.0.self_attn"}
        for kind, name, native in (("linear_attention", "linear_attn", "mamba"), ("full_attention", "self_attn", "self_attn")):
            if blocks(model, kind):
                first = blocks(model, kind)[0]
                rows[f"layers.{first}.{name}"] = f"model.layers.{first}.{native}"
        return rows

    def test_every_layer_is_renamed(self, model):
        linear, attention = blocks(model, "linear_attention"), blocks(model, "full_attention")
        assert sorted(linear + attention) == list(range(len(model.layers)))
        for i, layer in enumerate(model.layers):
            assert hasattr(layer, "linear_attn") == (i in linear)
            assert (getattr(layer, "self_attn", None) is not None) == (i in attention)
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

    def test_mlp_input_reaches_the_routed_and_the_shared_experts(self, model):
        layer = model.layers[0]
        if not layer._module.has_experts:
            pytest.skip("no routed experts: the shared expert is the whole feed-forward")
        written_mlp_input_reaches(model, layer, layer.block_sparse_moe, layer.mlp)


class HybridSuite(StateSpaceChecks, GraniteMoeHybridSuite):
    """The hybrid layouts: Mamba-2 blocks and NoPE attention blocks."""

    EXPECTED_UNAVAILABLE = {
        **{name: "no self_attn module" for name in VALUES if name.startswith("self_attn.")},
        **{name: "no " for name in NO_ROTARY},  # no self_attn module off the attention blocks, no rotary embedding on them
        **{f"linear_attn.{name}": "no linear_attn module" for name in LINEAR if name not in ("state", "states")},
        "linear_attn.state": "",   # missing off the Mamba-2 blocks, and unavailable on them
        "linear_attn.states": "",
        "linear_attn.set_state_after": "",
    }


class TestGraniteMoeHybrid(HybridSuite):
    REPO = REPO


class TestGraniteMoeHybridScaled(HybridSuite):
    """Granite's multipliers away from 1.0: the identity holds only on the scaled terms, on both block kinds."""

    REPO = built(test_granite._scaled_checkpoint, REPO)
    def test_mixer_contributions_are_scaled(self, model):
        ssd_block, attention_block = blocks(model, "linear_attention")[0], blocks(model, "full_attention")[0]
        with model.trace(PROMPT):
            ssd_raw = model.layers[ssd_block].linear_attn.output.save()
            ssd = model.layers[ssd_block].linear_attn.attention_output.save()
        with model.trace(PROMPT):
            attn_raw = model.layers[attention_block].self_attn.output[0].save()
            attn = model.layers[attention_block].self_attn.attention_output.save()
        scale = model.config.residual_multiplier
        torch.testing.assert_close(ssd, ssd_raw * scale)
        torch.testing.assert_close(attn, attn_raw * scale)

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
        torch.testing.assert_close(logits, raw / model.config.logits_scaling)
        torch.testing.assert_close(model.project_on_vocab(resid), logits)


class TestGraniteMoeHybridDense(HybridSuite):
    """``num_local_experts`` 0, as Granite-4.0-H-Micro: the shared expert is the whole feed-forward."""

    REPO = built(test_granite._scaled_checkpoint, REPO, num_local_experts=0)
    NEURONS_FROM_ACTIVATION = False  # input_linear holds the gate and up halves: the neurons are act(gate) * up

    def test_no_routed_experts(self, model):
        assert all(layer._module.block_sparse_moe is None for layer in model.layers)


def _attention_checkpoint(repo=REPO):
    """granite-4.0-350m's, -1b's and -micro's layout: every block attention with a rotary embedding
    (``position_embedding_type`` "rope"), no Mamba block, the feed-forward the shared MLP alone (``num_local_experts``
    0), Granite's multipliers away from 1.0. The tiny config otherwise, weights drawn for it."""
    config = json.load(open(os.path.join(cached_snapshot(repo), "config.json")))
    return patched_checkpoint(
        repo, draw=True, layer_types=["full_attention"] * config["num_hidden_layers"], position_embedding_type="rope",
        num_local_experts=0, residual_multiplier=0.22, logits_scaling=4.0, embedding_multiplier=3.0,
        attention_multiplier=config["num_attention_heads"] / config["hidden_size"],
    )


class TestGraniteMoeHybridAttention(GraniteMoeHybridSuite):
    """No Mamba block: the attention rotates its queries and keys, so the values before the rotary are served."""

    REPO = built(_attention_checkpoint)
    NEURONS_FROM_ACTIVATION = False  # input_linear holds the gate and up halves: the neurons are act(gate) * up

    def expected_values(self, model):
        """No block has a Mamba-2 mixer, so no ``linear_attn`` value is listed."""
        return {name for name in super().expected_values(model) if not name.startswith("linear_attn.")}

    def test_every_block_is_rotary_attention(self, model):
        assert model.config.position_embedding_type == "rope" and not blocks(model, "linear_attention")
        assert all(layer.self_attn.no_rotary() is None for layer in model.layers)
        assert all(layer._module.block_sparse_moe is None for layer in model.layers)
