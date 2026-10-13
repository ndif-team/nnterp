"""OPT, end to end: no MLP module."""

import glob
import json
import os
import tempfile

import pytest
import torch
from suite import FamilySuite, PROMPT, rows

from nnterp.families import opt


class TestOPT(FamilySuite):
    REPO = "hf-internal-testing/tiny-random-OPTForCausalLM"
    FAMILY = opt
    NATIVE = rows("model.decoder", "layers", "embed_tokens", "final_layer_norm", mlp=None, ln1="self_attn_layer_norm", ln2=None)
    MLP_NORM = "final_layer_norm"            # the block's own, native name (the decoder's has the same name)
    EXPECTED_UNAVAILABLE = {"mlp_input": "no mlp module"}

    def test_no_mlp_module(self, model):
        """No block has an MLP module, so `support()` lists no ``mlp`` value at all."""
        assert not hasattr(model.layers[0], "mlp")
        assert not any(name.startswith("mlp.") for name in model.support())
        assert not any(name.startswith("mlp.") for name in model.support(layer=0))

    def test_contribution_identity(self, model):
        """No MLP module, but an MLP path on the block: ``fc2``'s output is what the block adds."""
        parts = {}
        with model.trace(PROMPT):
            for i, layer in enumerate(model.layers):
                parts[i] = (layer.input.save(), layer.self_attn.attention_output.save(), layer.fc2.output.save(), layer.layer_output.save())
        for i, (x, attn, mlp, out) in parts.items():
            eps = torch.finfo(out.dtype).eps
            torch.testing.assert_close(x.float() + attn.float() + mlp.float(), out.float(), rtol=8 * eps, atol=8 * eps, msg=f"layer {i}")


def _post_norm_checkpoint(repo="hf-internal-testing/tiny-random-OPTForCausalLM"):
    """The tiny checkpoint with ``do_layer_norm_before`` false: OPT-350m's layout, the norms after the adds."""
    snapshot = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*"))[0]
    patched = tempfile.mkdtemp(prefix="opt-post-norm-")
    for name in os.listdir(snapshot):
        if name != "config.json":
            os.symlink(os.path.realpath(os.path.join(snapshot, name)), os.path.join(patched, name))
    config = json.load(open(os.path.join(snapshot, "config.json")))
    config["do_layer_norm_before"] = False
    json.dump(config, open(os.path.join(patched, "config.json"), "w"))
    return patched


class TestOPTPostNorm(TestOPT):
    """OPT-350m's layout: the norms follow the adds, so the sublayers read the stream itself, the identity does not
    hold, and transformers builds no final norm (no ``norm``, so no ``final_norm_output`` and no logit lens)."""

    REPO = _post_norm_checkpoint()
    NATIVE = {k: v for k, v in TestOPT.NATIVE.items() if k != "norm"}
    ATTENTION_NORM = None
    MLP_NORM = None
    MID_IDENTITY = False                      # layer_mid is the norm of the sum, not the sum
    EXPECTED_UNAVAILABLE = {
        "mlp_input": "no mlp module", "attention_norm_output": "a post-norm block", "mlp_norm_output": "a post-norm block",
        "final_norm_output": "no norm module",
    }

    def test_contribution_identity(self, model):
        pytest.skip("the norms follow the adds: input + attention_output + fc2.output is normed before it is the block's output")

    def test_trace_through_standard_names(self, model):
        pytest.skip("no final norm on this layout")

    def test_skip_layers_hands_the_stream_straight_through(self, model):
        pytest.skip("no final norm on this layout, so no logit lens to check the skip against")

    def test_project_on_vocab_is_the_logit_lens(self, model):
        with pytest.raises(AttributeError, match="norm"):
            model.project_on_vocab(torch.zeros(1, 1, model.hidden_size))

    def test_logits_are_the_models_output(self, model):
        """The lens half of the base test needs the final norm this layout lacks."""
        with model.trace(PROMPT):
            logits = model.logits.save()
            result = model.output.logits.save()
        assert torch.equal(logits, result) and logits.shape[-1] == model.vocab_size

    def test_layer_mid_is_the_normed_sum(self, model):
        assert not model.config.do_layer_norm_before
        with model.trace(PROMPT):
            mid = model.layers[0].layer_mid.save()
            raw = model.layers[0].self_attn_layer_norm.output.save()
        assert torch.equal(mid, raw.reshape(-1, raw.shape[-1]))
