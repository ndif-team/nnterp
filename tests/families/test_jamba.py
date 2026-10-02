"""Jamba, end to end: Mamba-1 blocks, attention blocks and a mixture of experts."""

import torch
from scan_suite import SelectiveScanSuite
from suite import LINEAR, VALUES, FamilySuite, PROMPT, rows

from nnter.families import jamba

ATTENTION_BLOCKS = (1,)   # ``attn_layer_offset`` 1 of ``attn_layer_period`` 8: block 1 of the two
MOE_BLOCKS = (1,)         # ``expert_layer_offset`` 1 of ``expert_layer_period`` 2

NATIVE = rows("model", "layers", "embed_tokens", "final_layernorm", mlp="feed_forward", ln2="pre_ff_layernorm")
del NATIVE["layers.0.self_attn"]
NATIVE.update({"layers.0.linear_attn": "model.layers.0.mamba", "layers.1.self_attn": "model.layers.1.self_attn"})


class TestJamba(SelectiveScanSuite, FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-JambaForCausalLM"
    FAMILY = jamba
    NATIVE = NATIVE
    REAL = ("ai21labs/Jamba-v0.1", 32, 4096, 8192)
    EXPECTED_UNAVAILABLE = {
        **{name: "no self_attn module" for name in VALUES if name.startswith("self_attn.")},
        **{f"linear_attn.{name}": "no linear_attn module" for name in LINEAR},
    }

    def test_every_layer_is_renamed(self, model):
        for i, layer in enumerate(model.layers):
            assert hasattr(layer, "linear_attn") == (i not in ATTENTION_BLOCKS)
            assert (getattr(layer, "self_attn", None) is not None) == (i in ATTENTION_BLOCKS)
            assert hasattr(layer, "mlp") and hasattr(layer, "input_layernorm") and hasattr(layer, "post_attention_layernorm")

    def test_layer_types_match_the_tree(self, model):
        kinds = ["linear_attention" if hasattr(layer, "linear_attn") else "full_attention" for layer in model.layers]
        assert kinds == list(model.config.layer_types)

    def test_mlp_is_dense_or_a_mixture_of_experts(self, model):
        experts = [i for i, layer in enumerate(model.layers) if hasattr(layer.mlp, "experts")]
        assert tuple(experts) == MOE_BLOCKS
        assert all(type(layer.mlp) is (jamba.Moe if i in MOE_BLOCKS else jamba.Mlp) for i, layer in enumerate(model.layers))

    def test_support_is_per_block_on_a_hybrid(self, model):
        support = model.support()
        scan = [i for i in range(len(model.layers)) if i not in ATTENTION_BLOCKS]
        assert set(support["self_attn.attention_probabilities"]) == set(scan)
        assert set(support["linear_attn.state_output"]) == set(ATTENTION_BLOCKS)
        assert support["layer_output"] is None and support["mlp.mlp_output"] is None

    def test_both_mixers_add_to_the_stream(self, model):
        """A Mamba block's scan and an attention block's attention are each the block's ``attention_output``, same layout."""
        with model.trace(PROMPT):
            scan = model.layers[0].linear_attn.attention_output.save()
            attn = model.layers[1].self_attn.attention_output.save()
        assert scan.shape == attn.shape and scan.shape[-1] == model.hidden_size
        assert not torch.equal(scan, torch.zeros_like(scan)) and not torch.equal(attn, torch.zeros_like(attn))
