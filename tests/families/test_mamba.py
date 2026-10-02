"""Mamba, end to end: a pure state-space model, every block a selective-scan mixer."""

import torch
from scan_suite import SelectiveScanSuite
from suite import FamilySuite, PROMPT, rows

from nnterp.families import mamba

NATIVE = rows("backbone", "layers", "embeddings", "norm_f", mlp=None, ln1="norm", ln2=None)
del NATIVE["layers.0.self_attn"]
NATIVE["layers.0.linear_attn"] = "backbone.layers.0.mixer"


class TestMamba(SelectiveScanSuite, FamilySuite):
    REPO = "hf-internal-testing/tiny-random-MambaForCausalLM"
    FAMILY = mamba
    NATIVE = NATIVE
    REAL = ("state-spaces/mamba-130m-hf", 24, 768, 1536)

    def test_every_block_is_a_mixer_only(self, model):
        for layer in model.layers:
            assert layer.linear_attn is layer.mixer and layer.input_layernorm is layer.norm
            assert getattr(layer, "self_attn", None) is None and getattr(layer, "mlp", None) is None

    def test_contribution_identity_without_an_mlp(self, model):
        """``input + attention_output == layer_output``: the mixer is the block's only contribution."""
        parts = {}
        with model.trace(PROMPT):
            for i, layer in enumerate(model.layers):
                parts[i] = (layer.input.save(), layer.linear_attn.attention_output.save(), layer.layer_output.save())
        for i, (x, mix, out) in parts.items():
            assert out.dtype == torch.float32  # residual_in_fp32: the block adds in float32
            torch.testing.assert_close(x.float() + mix.float(), out, msg=f"layer {i}")
