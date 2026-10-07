"""OLMo 3, end to end: post-norms only, sliding and full attention layers."""

import glob
import json
import os
import tempfile

import torch
from suite import FamilySuite, PROMPT, rows

from nnterp.families import olmo3


def _patched_checkpoint(repo="yujiepan/olmo-3-tiny-random"):
    """The tiny checkpoint with its config in transformers 5.17's per-layer-type rope form.

    The checkpoint's flat ``rope_parameters`` does not parse for a model with
    ``layer_types``, so the rewrite copies it to every layer type, YaRN on the
    sliding blocks too; the released Olmo 3 configs resolve to YaRN on the
    full-attention blocks and plain rotary on the sliding ones. The weights and
    tokenizer are symlinked, only ``config.json`` is rewritten. The family itself
    needs nothing.
    """
    snapshot = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*"))[0]
    patched = tempfile.mkdtemp(prefix="olmo3-")
    for name in os.listdir(snapshot):
        if name != "config.json":
            os.symlink(os.path.realpath(os.path.join(snapshot, name)), os.path.join(patched, name))
    config = json.load(open(os.path.join(snapshot, "config.json")))
    rope = config["rope_parameters"]
    if "rope_type" in rope:
        config["rope_parameters"] = {kind: dict(rope) for kind in set(config["layer_types"])}
    json.dump(config, open(os.path.join(patched, "config.json"), "w"))
    return patched


class TestOlmo3(FamilySuite):
    REPO = _patched_checkpoint()
    FAMILY = olmo3
    NATIVE = rows("model", "layers", "embed_tokens", "norm", ln1=None)
    ATTENTION_NORM = None
    MLP_NORM = None

    def test_contributions_are_the_post_norms(self, model):
        with model.trace(PROMPT):
            attn = model.layers[0].self_attn.attention_output.save()
            post_attn = model.layers[0].post_attention_layernorm.output.save()
        assert torch.equal(attn, post_attn)
