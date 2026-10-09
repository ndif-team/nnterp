"""Mamba-2, end to end: a pure state-space model, one norm and one SSD mixer per block."""

import torch
from ssd import StateSpaceChecks
from suite import LINEAR, FamilySuite, PROMPT

from nnterp.families import mamba2


class TestMamba2(StateSpaceChecks, FamilySuite):
    REPO = "yujiepan/mamba2-tiny-random"
    DECODE_MASK_COVERS_CACHE = False  # a decode step is fed the new token's mask only
    FAMILY = mamba2
    ATTENTION_NORM = "norm"                # the block norm keeps its native name
    NATIVE = {
        "embed_tokens": "backbone.embeddings",
        "layers": "backbone.layers",
        "layers.0.linear_attn": "backbone.layers.0.mixer",
        "norm": "backbone.norm_f",
        "lm_head": "lm_head",
    }
    EXPECTED_UNAVAILABLE = {
        "linear_attn.state": "one tensor per call",
        "linear_attn.states": "chunk_per_token",
        "linear_attn.set_state_after": "one cumulative step",
    }

    def test_block_is_the_mixer_alone(self, model):
        """No attention, no MLP: ``support()`` lists neither, and the mixer is the block's one contribution."""
        support = model.support()
        assert not any(name.startswith(("self_attn.", "mlp.")) for name in support)
        assert {f"linear_attn.{name}" for name in LINEAR} <= set(support)
        parts = {}
        with model.trace(PROMPT):
            for i, layer in enumerate(model.layers):
                parts[i] = (layer.input.save(), layer.linear_attn.attention_output.save(), layer.layer_output.save())
        for i, (x, mix, out) in parts.items():
            eps = torch.finfo(out.dtype).eps
            torch.testing.assert_close(x.float() + mix.float(), out.float(), rtol=8 * eps, atol=8 * eps, msg=f"layer {i}")

    def test_only_the_block_norm_is_aliased(self, model):
        """``norm`` is the block's; the mixer's own gated ``norm`` keeps its native name only."""
        layer = model.layers[0]
        assert "input_layernorm" not in layer.__dict__ and "input_layernorm" not in layer.linear_attn._aliases
        assert layer._aliases == {"linear_attn": "mixer"}

    def test_sizes_are_the_mixers(self, model):
        m = model.layers[0].linear_attn._module
        assert model.num_layers == len(model.layers) == model.config.num_hidden_layers
        assert model.num_heads == model.num_kv_heads == m.num_heads and model.head_dim == m.head_dim
        assert model.intermediate_size == m.out_proj.in_features == m.num_heads * m.head_dim
        assert model.hidden_size == m.out_proj.out_features
