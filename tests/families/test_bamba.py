"""Bamba, end to end: Llama's block with a Mamba-2 mixer or attention."""

from ssd import StateSpaceChecks
from suite import LINEAR, LLAMA_ROWS, VALUES, FamilySuite, blocks

from nnterp.families import bamba


class TestBamba(StateSpaceChecks, FamilySuite):
    """The tiny checkpoint: attention at blocks 1 and 3 (``attn_layer_indices``), Mamba-2 at 0 and 2."""

    REPO = "hf-tiny-v2/tiny-random-BambaForCausalLM"
    FAMILY = bamba
    NATIVE = {
        **{k: v for k, v in LLAMA_ROWS.items() if k != "layers.0.self_attn"},
        "norm": "model.final_layernorm",
        "layers.0.mlp": "model.layers.0.feed_forward",
        "layers.0.post_attention_layernorm": "model.layers.0.pre_ff_layernorm",
        "layers.0.linear_attn": "model.layers.0.mamba",
    }
    EXPECTED_UNAVAILABLE = {
        **{name: "no self_attn module" for name in VALUES if name.startswith("self_attn.")},
        **{f"linear_attn.{name}": "no linear_attn module" for name in LINEAR if name not in ("state", "states")},
        "linear_attn.state": "",   # missing off the Mamba-2 blocks, and unavailable on them
        "linear_attn.states": "",
        "linear_attn.set_state_after": "",
    }

    def native(self, model):
        """`NATIVE` with the first attention block where ``attn_layer_indices`` puts it (block 1 on the tiny checkpoint)."""
        first = blocks(model, "full_attention")[0]
        return {**self.NATIVE, f"layers.{first}.self_attn": f"model.layers.{first}.self_attn"}

    def test_every_layer_is_renamed(self, model):
        ssd, attention = blocks(model, "linear_attention"), blocks(model, "full_attention")
        assert ssd and attention and sorted(ssd + attention) == list(range(len(model.layers)))
        for i, layer in enumerate(model.layers):
            assert hasattr(layer, "linear_attn") == (i in ssd)
            assert (getattr(layer, "self_attn", None) is not None) == (i in attention)
            assert layer.mlp is layer.feed_forward and layer.post_attention_layernorm is layer.pre_ff_layernorm

    def test_support_is_per_block(self, model):
        support = model.support()
        assert set(support["self_attn.attention_probabilities"]) == set(blocks(model, "linear_attention"))
        assert set(support["linear_attn.state_output"]) == set(blocks(model, "full_attention"))
        assert support["layer_output"] is None and support["mlp.mlp_output"] is None
