"""Doge, end to end: a gated residual (the stream is scaled, the contributions are not) and a dynamic mask."""

import glob
import os
import tempfile

import torch
from suite import FamilySuite, LLAMA_ROWS, PROMPT

from nnter.families import doge

REPO = "hf-tiny-v2/tiny-random-DogeForCausalLM"


def _gated_checkpoint(repo=REPO):
    """The tiny checkpoint with its residual gates set away from one.

    The tiny checkpoint's ``input_residual`` and ``post_attention_residual`` are
    their initial ones, where the gated identity is the plain one. The copy draws
    both from ``[0.5, 1.5)`` with a fixed seed and writes them into a copy of the
    weights, once per snapshot; everything else is symlinked.
    """
    from safetensors.torch import load_file, save_file

    snapshot = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*"))[0]
    patched = os.path.join(tempfile.gettempdir(), f"nnter-doge-gated-{os.path.basename(snapshot)}")
    if os.path.exists(os.path.join(patched, "model.safetensors")):
        return patched
    os.makedirs(patched, exist_ok=True)
    weights = load_file(os.path.join(snapshot, "model.safetensors"))
    generator = torch.Generator().manual_seed(0)
    for name, tensor in weights.items():
        if name.endswith(("input_residual", "post_attention_residual")):
            weights[name] = torch.rand(tensor.shape, generator=generator, dtype=tensor.dtype) + 0.5
    save_file(weights, os.path.join(patched, "model.safetensors"), metadata={"format": "pt"})
    for name in os.listdir(snapshot):
        target = os.path.join(patched, name)
        if name != "model.safetensors" and not os.path.exists(target):
            os.symlink(os.path.realpath(os.path.join(snapshot, name)), target)
    return patched


def gated_identity(model):
    """``post_attention_residual * (input_residual * input + attention_output) + mlp_output == layer_output`` on every block."""
    read = []  # bound outside: a name bound inside the block does not survive it
    with model.trace(PROMPT):
        for layer in model.layers:
            read.append((layer.input.save(), layer.self_attn.attention_output.save(), layer.mlp.mlp_output.save(), layer.layer_output.save()))
    for layer, (x, attn, mlp, out) in zip(model.layers, read):
        block = layer._module
        torch.testing.assert_close(block.post_attention_residual * (block.input_residual * x + attn) + mlp, out)


class TestDoge(FamilySuite):
    REPO = REPO
    FAMILY = doge
    NATIVE = LLAMA_ROWS

    def test_gates_are_one_on_the_tiny_checkpoint(self, model):
        for layer in model.layers:
            assert torch.equal(layer._module.input_residual, torch.ones_like(layer._module.input_residual))
            assert torch.equal(layer._module.post_attention_residual, torch.ones_like(layer._module.post_attention_residual))

    def test_gated_identity(self, model):
        gated_identity(model)


class TestDogeGated(FamilySuite):
    """Gates away from one: the contributions are still the modules' outputs, and the identity carries the gates."""

    REPO = _gated_checkpoint()
    FAMILY = doge
    NATIVE = LLAMA_ROWS

    def test_contribution_identity(self, model):
        """The suite's identity with the block's gates on the residual."""
        gated_identity(model)

    def test_contributions_are_the_module_outputs(self, model):
        with model.trace(PROMPT):
            attn_raw = model.layers[0].self_attn.output[0].save()
            attn = model.layers[0].self_attn.attention_output.save()
            mlp_raw = model.layers[0].mlp.output.save()
            mlp = model.layers[0].mlp.mlp_output.save()
        assert torch.equal(attn, attn_raw) and torch.equal(mlp, mlp_raw)

    def test_plain_identity_does_not_hold(self, model):
        with model.trace(PROMPT):
            x = model.layers[0].input.save()
            attn = model.layers[0].self_attn.attention_output.save()
            mlp = model.layers[0].mlp.mlp_output.save()
            out = model.layers[0].layer_output.save()
        assert not torch.allclose(x + attn + mlp, out)
