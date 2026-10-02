"""Nemotron-H, end to end: one mixer per block, a Mamba-2 mixer, attention or a mixture of experts."""

import glob
import os
import tempfile

import torch
from safetensors.torch import load_file, save_file
from ssd import StateSpaceChecks
from suite import LINEAR, VALUES, FamilySuite, PROMPT

from nnter.families import nemotron_h

KINDS = {"linear_attention": "linear_attn", "full_attention": "self_attn", "moe": "mlp", "mlp": "mlp"}


def _patched_checkpoint(repo="hf-tiny-v2/tiny-random-NemotronHForCausalLM"):
    """The tiny checkpoint with its embedding under the name transformers 5.17 loads.

    It stores ``backbone.embedding.weight``; the model's module is
    ``model.embeddings``, so a load would leave the embedding randomly
    initialized and two loads would differ. The weights are rewritten with
    the key renamed; everything else is symlinked. The family needs nothing.
    """
    snapshot = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*"))[0]
    patched = tempfile.mkdtemp(prefix="nemotron_h-")
    for name in os.listdir(snapshot):
        if name != "model.safetensors":
            os.symlink(os.path.realpath(os.path.join(snapshot, name)), os.path.join(patched, name))
    weights = load_file(os.path.join(snapshot, "model.safetensors"))
    weights["backbone.embeddings.weight"] = weights.pop("backbone.embedding.weight")
    save_file(weights, os.path.join(patched, "model.safetensors"), metadata={"format": "pt"})
    return patched


class TestNemotronH(StateSpaceChecks, FamilySuite):
    REPO = _patched_checkpoint()
    FAMILY = nemotron_h
    ATTENTION_NORM = "norm"                # the block norm keeps its native name
    # ``layers_block_type``: Mamba-2, MoE, Mamba-2, attention, MoE
    NATIVE = {
        "embed_tokens": "model.embeddings",
        "layers": "model.layers",
        "layers.0.linear_attn": "model.layers.0.mixer",
        "layers.1.mlp": "model.layers.1.mixer",
        "layers.3.self_attn": "model.layers.3.mixer",
        "norm": "model.norm_f",
        "lm_head": "lm_head",
    }
    MLP_WIDTH_KEY = "moe_intermediate_size"
    EXPECTED_UNAVAILABLE = {
        **{name: "no self_attn module" for name in VALUES if name.startswith("self_attn.")},
        **{f"linear_attn.{name}": "no linear_attn module" for name in LINEAR if name not in ("state", "states")},
        "linear_attn.state": "",   # missing off the Mamba-2 blocks, and unavailable on them
        "linear_attn.states": "",
        "linear_attn.set_state_after": "",
        "mlp.mlp_output": "no mlp module",
    }

    def test_every_layer_is_renamed(self, model):
        """Each block's ``mixer`` answers to the standard name of what it is, and to nothing else."""
        for layer, kind in zip(model.layers, model.config.layers_block_type):
            name = KINDS[kind]
            assert getattr(layer, name) is layer.mixer, (layer.path, kind)
            assert all(getattr(layer, other, None) is None for other in set(KINDS.values()) - {name}), (layer.path, kind)
            assert layer._aliases == {name: "mixer"}

    def test_support_is_per_block(self, model):
        support = model.support()
        kinds = model.config.layers_block_type
        assert set(support["self_attn.attention_probabilities"]) == {i for i, k in enumerate(kinds) if k != "full_attention"}
        assert set(support["linear_attn.state_output"]) == {i for i, k in enumerate(kinds) if k != "linear_attention"}
        assert set(support["mlp.mlp_output"]) == {i for i, k in enumerate(kinds) if k not in ("moe", "mlp")}
        assert support["layer_output"] is None

    def test_one_contribution_per_block(self, model):
        """``input + <the block's one sublayer> == layer_output`` on every block."""
        parts = {}
        with model.trace(PROMPT):
            for i, (layer, kind) in enumerate(zip(model.layers, model.config.layers_block_type)):
                host = getattr(layer, KINDS[kind])
                value = "mlp_output" if KINDS[kind] == "mlp" else "attention_output"
                parts[i] = (layer.input.save(), getattr(host, value).save(), layer.layer_output.save())
        for i, (x, added, out) in parts.items():
            torch.testing.assert_close(x + added, out, msg=f"layer {i}")

    def test_aliases_survive_dispatch(self):
        """The block binds its aliases from the mixer's class at build and again when real weights arrive."""
        from nnter import StandardizedTransformer

        lazy = StandardizedTransformer(self.REPO)
        assert lazy.layers[3].self_attn is lazy.layers[3].mixer
        with lazy.trace(PROMPT):
            out = lazy.layers[3].self_attn.attention_output.save()
        assert lazy.layers[3].self_attn is lazy.layers[3].mixer and out.shape[-1] == lazy.hidden_size
