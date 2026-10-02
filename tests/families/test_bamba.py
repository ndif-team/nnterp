"""Bamba, end to end: Llama's block with a Mamba-2 mixer or attention."""

from ssd import StateSpaceChecks
from suite import LINEAR, LLAMA_ROWS, VALUES, FamilySuite

from nnter.families import bamba

SSD_BLOCKS = (0, 2)       # ``attn_layer_indices`` = [1, 3]
ATTENTION_BLOCKS = (1, 3)


class TestBamba(StateSpaceChecks, FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-BambaForCausalLM"
    FAMILY = bamba
    NATIVE = {
        **{k: v for k, v in LLAMA_ROWS.items() if k != "layers.0.self_attn"},
        "norm": "model.final_layernorm",
        "layers.0.mlp": "model.layers.0.feed_forward",
        "layers.0.post_attention_layernorm": "model.layers.0.pre_ff_layernorm",
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
            assert layer.mlp is layer.feed_forward and layer.post_attention_layernorm is layer.pre_ff_layernorm

    def test_support_is_per_block(self, model):
        support = model.support()
        assert set(support["self_attn.attention_probabilities"]) == set(SSD_BLOCKS)
        assert set(support["linear_attn.state_output"]) == set(ATTENTION_BLOCKS)
        assert support["layer_output"] is None and support["mlp.mlp_output"] is None
