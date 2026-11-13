"""Jamba, end to end: Mamba-1 blocks, attention blocks and a mixture of experts."""

import torch
from scan_suite import SelectiveScanSuite
from suite import LINEAR, NO_ROTARY, VALUES, FamilySuite, PROMPT, blocks, patched_checkpoint, rows, built

from nnterp.families import jamba

REPO = "hf-tiny-v2/tiny-random-JambaForCausalLM"
NATIVE = rows("model", "layers", "embed_tokens", "final_layernorm", mlp="feed_forward", ln2="pre_ff_layernorm")
del NATIVE["layers.0.self_attn"]
NATIVE.update({"layers.0.linear_attn": "model.layers.0.mamba"})


def moe_blocks(model):
    """The blocks whose feed-forward is a mixture (``layers_num_experts`` above 1): block 1 of the tiny checkpoint's
    two (``expert_layer_offset`` 1 of ``expert_layer_period`` 2), none on Jamba2-3B and Jamba-Reasoning-3B."""
    return [i for i, n in enumerate(model.config.layers_num_experts) if n > 1]


class TestJamba(SelectiveScanSuite, FamilySuite):
    """The tiny checkpoint: attention at block 1 of two (``attn_layer_offset`` 1 of ``attn_layer_period`` 8), a
    mixture at block 1. The blocks of each kind are read from the config."""

    REPO = REPO
    FAMILY = jamba
    NATIVE = NATIVE
    REAL = ("ai21labs/Jamba-v0.1", 32, 4096, 8192)
    EXPECTED_UNAVAILABLE = {
        **{name: "no self_attn module" for name in VALUES if name.startswith("self_attn.")},
        **{name: "no " for name in NO_ROTARY},  # no self_attn module off the attention blocks, no rotary embedding on them
        **{f"linear_attn.{name}": "no linear_attn module" for name in LINEAR},
    }

    def native(self, model):
        """`NATIVE` with the first attention block where the config puts it."""
        first = blocks(model, "full_attention")[0]
        return {**self.NATIVE, f"layers.{first}.self_attn": f"model.layers.{first}.self_attn"}

    def test_every_layer_is_renamed(self, model):
        attention = blocks(model, "full_attention")
        assert attention and sorted(attention + blocks(model, "linear_attention")) == list(range(len(model.layers)))
        for i, layer in enumerate(model.layers):
            assert hasattr(layer, "linear_attn") == (i not in attention)
            assert (getattr(layer, "self_attn", None) is not None) == (i in attention)
            assert hasattr(layer, "mlp") and hasattr(layer, "input_layernorm") and hasattr(layer, "post_attention_layernorm")

    def test_layer_types_match_the_tree(self, model):
        kinds = ["linear_attention" if hasattr(layer, "linear_attn") else "full_attention" for layer in model.layers]
        assert kinds == list(model.config.layer_types)

    def test_mlp_is_dense_or_a_mixture_of_experts(self, model):
        experts = [i for i, layer in enumerate(model.layers) if hasattr(layer.mlp, "experts")]
        assert experts == moe_blocks(model)
        assert all(type(layer.mlp) is (jamba.Moe if i in experts else jamba.Mlp) for i, layer in enumerate(model.layers))

    def test_support_is_per_block_on_a_hybrid(self, model):
        support = model.support()
        assert set(support["self_attn.attention_probabilities"]) == set(blocks(model, "linear_attention"))
        assert set(support["linear_attn.state_output"]) == set(blocks(model, "full_attention"))
        assert support["layer_output"] is None and support["mlp.mlp_output"] is None

    def test_both_mixers_add_to_the_stream(self, model):
        """A Mamba block's scan and an attention block's attention are each the block's ``attention_output``, same layout."""
        first = blocks(model, "full_attention")[0]
        with model.trace(PROMPT):
            scan = model.layers[0].linear_attn.attention_output.save()
            attn = model.layers[first].self_attn.attention_output.save()
        assert scan.shape == attn.shape and scan.shape[-1] == model.hidden_size
        assert not torch.equal(scan, torch.zeros_like(scan)) and not torch.equal(attn, torch.zeros_like(attn))


class TestJambaDense(TestJamba):
    """Jamba2-3B's and Jamba-Reasoning-3B's layout: ``num_experts`` 1, so every feed-forward is a dense MLP and no
    block has a mixture. The tiny config otherwise, weights drawn for it."""

    REPO = built(patched_checkpoint, REPO, draw=True, num_experts=1)
    def test_no_block_has_a_mixture(self, model):
        assert model.config.num_experts == 1 and moe_blocks(model) == []
        assert all(type(layer.mlp) is jamba.Mlp for layer in model.layers)
