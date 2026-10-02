"""ZAYA, end to end: residual scaling modules that scale both the contribution and the stream."""

import glob
import os
import tempfile

import torch
from suite import FamilySuite, LLAMA_ROWS, PROMPT

from nnterp.families import zaya

REPO = "hf-tiny-v2/tiny-random-ZayaForCausalLM"


def _patched_checkpoint(merges, repo=REPO):
    """The tiny checkpoint with its key temperatures set to one and, with ``merges``, its residual merges moved.

    The tiny checkpoint's ``qk_norm.temp`` is zero, its initial value, which
    zeroes every key: the pattern is uniform, the queries are inert and both
    blocks' patterns are equal, so every copy sets it to one. With ``merges``,
    every residual scaling module's scales are drawn from ``[0.5, 1.5)`` and its
    biases from ``[-0.1, 0.1)``: the tiny checkpoint's are at their initial values
    (scales one, biases zero), where the identity is the plain one. Written once
    per snapshot, seeded; everything else is symlinked.
    """
    from safetensors.torch import load_file, save_file

    snapshot = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*"))[0]
    patched = os.path.join(tempfile.gettempdir(), f"nnterp-zaya-{'merged' if merges else 'temp'}-{os.path.basename(snapshot)}")
    if os.path.exists(os.path.join(patched, "model.safetensors")):
        return patched
    os.makedirs(patched, exist_ok=True)
    weights = load_file(os.path.join(snapshot, "model.safetensors"))
    generator = torch.Generator().manual_seed(0)
    for name, tensor in weights.items():
        if name.endswith("qk_norm.temp"):
            weights[name] = torch.ones_like(tensor)
        elif not merges:
            continue
        elif "residual_scale." in name and name.endswith("_scale"):
            weights[name] = torch.rand(tensor.shape, generator=generator, dtype=tensor.dtype) + 0.5
        elif "residual_scale." in name and name.endswith("_bias"):
            weights[name] = (torch.rand(tensor.shape, generator=generator, dtype=tensor.dtype) - 0.5) / 5
    save_file(weights, os.path.join(patched, "model.safetensors"), metadata={"format": "pt"})
    for name in os.listdir(snapshot):
        target = os.path.join(patched, name)
        if name != "model.safetensors" and not os.path.exists(target):
            os.symlink(os.path.realpath(os.path.join(snapshot, name)), target)
    return patched


def merged_identity(model):
    """Each block's output from its input and the two contributions, through its merges' residual scales and biases."""
    read = []  # bound outside: a name bound inside the block does not survive it
    with model.trace(PROMPT):
        for layer in model.layers:
            read.append((layer.input.save(), layer.self_attn.attention_output.save(), layer.mlp.mlp_output.save(), layer.layer_output.save()))
    for layer, (x, attn, mlp, out) in zip(model.layers, read):
        first, second = layer._module.post_attention_residual_scale, layer._module.post_mlp_residual_scale
        h = (x + first.residual_bias) * first.residual_scale + attn
        torch.testing.assert_close((h + second.residual_bias) * second.residual_scale + mlp, out)


class TestZaya(FamilySuite):
    """Key temperatures of one; the merges at the tiny checkpoint's initial values."""

    REPO = _patched_checkpoint(merges=False)
    FAMILY = zaya
    NATIVE = LLAMA_ROWS
    MLP_WIDTH_KEY = "moe_intermediate_size"   # every block is a mixture
    ROUTER_EXTRA_CLASSES = 1  # router_logits' last column is the skip class

    def test_merged_identity(self, model):
        merged_identity(model)


class TestZayaMerged(FamilySuite):
    """Merges away from their initial values: the contributions are the scaled terms, the stream is rescaled too."""

    REPO = _patched_checkpoint(merges=True)
    FAMILY = zaya
    NATIVE = LLAMA_ROWS
    MLP_WIDTH_KEY = "moe_intermediate_size"
    ROUTER_EXTRA_CLASSES = 1  # router_logits' last column is the skip class

    def test_contribution_identity(self, model):
        """The suite's identity through each merge's residual scale and bias."""
        merged_identity(model)

    def test_contributions_are_the_scaled_module_outputs(self, model):
        merge = model.layers[0]._module.post_attention_residual_scale
        with model.trace(PROMPT):
            attn_raw = model.layers[0].self_attn.output[0].save()
            attn = model.layers[0].self_attn.attention_output.save()
        torch.testing.assert_close(attn, (attn_raw + merge.hidden_states_bias) * merge.hidden_states_scale)

    def test_a_read_leaves_the_forward_untouched(self, model):
        with model.trace(PROMPT):
            clean = model.logits.save()
        with model.trace(PROMPT):
            for layer in model.layers:
                layer.self_attn.attention_output.save()
                layer.mlp.mlp_output.save()
            read = model.logits.save()
        assert torch.equal(clean, read)

    def test_assignment_and_in_place_edits_set_what_the_block_adds(self, model):
        merge = model.layers[0]._module.post_attention_residual_scale
        with model.trace(PROMPT):
            layer = model.layers[0]
            x = layer.input.save()
            layer.self_attn.attention_output = torch.ones_like(x)
            mid = layer.post_attention_layernorm.input.save()
        torch.testing.assert_close(mid, (x + merge.residual_bias) * merge.residual_scale + 1)
        with model.trace(PROMPT):
            layer = model.layers[0]
            x = layer.input.save()
            attn = layer.self_attn.attention_output
            attn[:] = 0
            kept = attn.save()
            mid = layer.post_attention_layernorm.input.save()
        assert torch.equal(kept, torch.zeros_like(kept))
        torch.testing.assert_close(mid, (x + merge.residual_bias) * merge.residual_scale)
