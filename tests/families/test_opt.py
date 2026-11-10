"""OPT, end to end: no MLP module."""

import glob
import json
import os
import tempfile

import pytest
import torch
from suite import NO_ROTARY, FamilySuite, PROMPT, rows

from nnterp.families import opt


class TestOPT(FamilySuite):
    REPO = "hf-internal-testing/tiny-random-OPTForCausalLM"
    FAMILY = opt
    NATIVE = rows("model.decoder", "layers", "embed_tokens", "final_layer_norm", mlp=None, ln1="self_attn_layer_norm", ln2=None)
    MLP_NORM = "final_layer_norm"            # the block's own, native name (the decoder's has the same name)
    EXPECTED_UNAVAILABLE = {**NO_ROTARY, "mlp_input": "no mlp module"}

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


def _projected_checkpoint(do_layer_norm_before, repo="hf-internal-testing/tiny-random-OPTForCausalLM"):
    """The tiny config with ``word_embed_proj_dim`` half of ``hidden_size``, saved with random weights: transformers
    builds ``project_in`` / ``project_out`` around the stack, as on OPT-350m (512 <-> 1024), and the embedding and
    the head are the narrower width."""
    from transformers import AutoConfig, OPTForCausalLM

    snapshot = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*"))[0]
    patched = tempfile.mkdtemp(prefix="opt-projected-")
    for name in os.listdir(snapshot):
        if name != "config.json" and not name.endswith((".safetensors", ".bin")):
            os.symlink(os.path.realpath(os.path.join(snapshot, name)), os.path.join(patched, name))
    config = AutoConfig.from_pretrained(snapshot)
    config.do_layer_norm_before = do_layer_norm_before
    config.word_embed_proj_dim = config.hidden_size // 2
    torch.manual_seed(0)
    OPTForCausalLM(config).save_pretrained(patched)
    return patched


class TestOPTPostNorm(TestOPT):
    """The norms follow the adds (OPT-350m's ``do_layer_norm_before`` false), so the sublayers read the stream itself,
    the identity does not hold, and transformers builds no final norm: no ``norm`` and no ``final_norm_output``, and
    the logit lens is ``lm_head`` alone."""

    REPO = _post_norm_checkpoint()
    NATIVE = {k: v for k, v in TestOPT.NATIVE.items() if k != "norm"}
    ATTENTION_NORM = None
    MLP_NORM = None
    MID_IDENTITY = False                      # layer_mid is the norm of the sum, not the sum
    EXPECTED_UNAVAILABLE = {
        **NO_ROTARY, "mlp_input": "no mlp module", "attention_norm_output": "a post-norm block", "mlp_norm_output": "a post-norm block",
        "final_norm_output": "no norm module",
    }

    def test_contribution_identity(self, model):
        pytest.skip("the norms follow the adds: input + attention_output + fc2.output is normed before it is the block's output")

    def test_trace_through_standard_names(self, model):
        pytest.skip("no final norm on this layout")

    def test_no_norm_module(self, model):
        """With no final norm, ``norm`` is a module the model lacks: reading it raises as for any, and `support` names the
        value read there as unavailable; the lens reads the stream without it."""
        assert model.model.decoder.final_layer_norm is None
        with pytest.raises(AttributeError, match="norm"):
            model.norm
        assert model.support()["final_norm_output"] == "no norm module on this model"

    def test_layer_mid_is_the_normed_sum(self, model):
        assert not model.config.do_layer_norm_before
        with model.trace(PROMPT):
            mid = model.layers[0].layer_mid.save()
            raw = model.layers[0].self_attn_layer_norm.output.save()
        assert torch.equal(mid, raw.reshape(-1, raw.shape[-1]))


class ProjectedLayout:
    """``project_in`` / ``project_out`` around the stack: ``token_embeddings`` is the embedding's output, before
    ``project_in``, so it is ``word_embed_proj_dim`` wide, not the ``hidden_size`` its ``Residual`` layout names."""

    def test_token_embeddings_are_the_embedding_output(self, model):
        pytest.skip("token_embeddings is word_embed_proj_dim wide on this layout, not hidden_size")

    def test_values_match_their_annotations(self, model):
        pytest.skip("token_embeddings is word_embed_proj_dim wide on this layout, not the hidden_size of its Residual layout")

    def test_lens_reads_project_out(self, model):
        """The lens is the decoder's own path: ``project_out`` after any final norm, then ``lm_head``."""
        decoder = model.model.decoder
        with model.trace(PROMPT):
            resid = model.layers[-1].layer_output.save()
            logits = model.logits.save()
        normed = resid if decoder.final_layer_norm is None else decoder.final_layer_norm(resid)
        torch.testing.assert_close(model.project_on_vocab(resid), model.lm_head(decoder.project_out(normed)))
        torch.testing.assert_close(model.project_on_vocab(resid), logits)


class TestOPT350mLayout(ProjectedLayout, TestOPTPostNorm):
    """OPT-350m's whole layout: post-norm blocks, no final norm, and ``project_in`` / ``project_out`` between the
    embedding width and the stream's. The decoder makes its logits as ``lm_head(project_out(h))``, so the lens does."""

    REPO = _projected_checkpoint(do_layer_norm_before=False)


class TestOPTProjected(ProjectedLayout, TestOPT):
    """A pre-norm OPT with ``project_out``: the decoder norms the stream, projects it, and the head reads that."""

    REPO = _projected_checkpoint(do_layer_norm_before=True)
