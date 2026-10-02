"""EXAONE 4.0, end to end: post-norms only, sliding-window layers with the rotary and full layers without."""

import glob
import json
import os
import tempfile

import torch
from suite import FamilySuite, PROMPT, rows

from nnterp.families import exaone4


class TestExaone4(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-Exaone4ForCausalLM"
    FAMILY = exaone4
    NATIVE = rows("model", "layers", "embed_tokens", "norm", ln1=None)
    ATTENTION_NORM = None                    # post-norms only: the block input enters the attention
    MLP_NORM = None                          # ... and the MLP takes the residual stream after the attention add

    def test_contributions_are_the_post_norms(self, model):
        with model.trace(PROMPT):
            attn = model.layers[0].self_attn.attention_output.save()
            post_attn = model.layers[0].post_attention_layernorm.output.save()
            mlp = model.layers[0].mlp.mlp_output.save()
            post_mlp = model.layers[0].post_feedforward_layernorm.output.save()
        assert torch.equal(attn, post_attn) and torch.equal(mlp, post_mlp)


def _hybrid_checkpoint(repo="hf-tiny-v2/tiny-random-Exaone4ForCausalLM"):
    """The tiny checkpoint with its second layer switched to full attention, as in EXAONE-4.0-32B's ``LLLG``.

    Every layer of the tiny checkpoint is a sliding-window one, so the full
    layers' NoPE branch (no rotary) never runs. The weights and tokenizer are
    symlinked, only ``config.json`` is rewritten.
    """
    snapshot = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*"))[0]
    patched = tempfile.mkdtemp(prefix="exaone4-hybrid-")
    for name in os.listdir(snapshot):
        if name != "config.json":
            os.symlink(os.path.realpath(os.path.join(snapshot, name)), os.path.join(patched, name))
    config = json.load(open(os.path.join(snapshot, "config.json")))
    config["layer_types"] = ["sliding_attention", "full_attention"]
    json.dump(config, open(os.path.join(patched, "config.json"), "w"))
    return patched


class TestExaone4Hybrid(FamilySuite):
    """A sliding-window layer and a full-attention layer that skips the rotary."""

    REPO = _hybrid_checkpoint()
    FAMILY = exaone4
    NATIVE = rows("model", "layers", "embed_tokens", "norm", ln1=None)
    ATTENTION_NORM = None
    MLP_NORM = None

    def test_the_full_layer_skips_the_rotary(self, model):
        sliding, full = (layer.self_attn._module for layer in model.layers)
        assert sliding.is_sliding and not full.is_sliding and full.sliding_window is not None

    def test_the_full_layer_has_no_rotary(self, model):
        """The keys reach the interface straight from the k norm on the full layer, rotated on the sliding one."""
        with model.trace(PROMPT):
            normed_sliding = model.layers[0].self_attn.k_norm.output.save()
            keys_sliding = model.layers[0].self_attn.attention_keys.save()
            normed_full = model.layers[1].self_attn.k_norm.output.save()
            keys_full = model.layers[1].self_attn.attention_keys.save()
        assert torch.equal(keys_full, normed_full)
        assert not torch.equal(keys_sliding, normed_sliding)
