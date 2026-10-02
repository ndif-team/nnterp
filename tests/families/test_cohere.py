"""Cohere (Command-R), end to end: a parallel block, and logits scaled after the head."""

import glob
import json
import os
import tempfile

import torch
from suite import FamilySuite, PROMPT, rows

from nnter.families import cohere


class TestCohere(FamilySuite):
    REPO = "trl-internal-testing/tiny-CohereForCausalLM"
    FAMILY = cohere
    NATIVE = rows("model", "layers", "embed_tokens", "norm", ln2=None)
    MLP_NORM = "input_layernorm"  # parallel: one norm feeds both sublayers

    def test_logits_are_the_head_times_logit_scale(self, model):
        scale = model.config.logit_scale
        assert scale != 1  # the tiny checkpoint's 0.125, so the hook is exercised
        with model.trace(PROMPT):
            resid = model.layers[-1].layer_output.save()
            raw = model.lm_head.output.save()
            logits = model.logits.save()
        torch.testing.assert_close(logits, raw * scale)
        torch.testing.assert_close(model.project_on_vocab(resid), logits)
        assert not torch.allclose(model.lm_head(model.norm(resid)), logits)  # the head alone is not the logits


def _qk_norm_checkpoint(repo="trl-internal-testing/tiny-CohereForCausalLM"):
    """The tiny checkpoint with ``use_qk_norm`` on (Command-R+'s layout); the new per-head norms load at their init."""
    snapshot = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*"))[0]
    patched = tempfile.mkdtemp(prefix="cohere-qk-norm-")
    for name in os.listdir(snapshot):
        if name != "config.json":
            os.symlink(os.path.realpath(os.path.join(snapshot, name)), os.path.join(patched, name))
    config = json.load(open(os.path.join(snapshot, "config.json")))
    config["use_qk_norm"] = True
    json.dump(config, open(os.path.join(patched, "config.json"), "w"))
    return patched


class TestCohereQkNorm(FamilySuite):
    """Command-R+'s per-head ``q_norm`` / ``k_norm`` run before the rotary; the interface still receives the queries."""

    REPO = _qk_norm_checkpoint()
    FAMILY = cohere
    NATIVE = rows("model", "layers", "embed_tokens", "norm", ln2=None)
    MLP_NORM = "input_layernorm"

    def test_the_norms_are_there(self, model):
        assert model.layers[0].self_attn._module.use_qk_norm
