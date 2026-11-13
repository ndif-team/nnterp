"""Doge, end to end: a gated residual (the stream is scaled, the contributions are not) and a dynamic mask."""

import os
import tempfile

import pytest
import torch
import transformers
from packaging.version import Version
from suite import FamilySuite, LLAMA_ROWS, PROMPT, cached_snapshot, built, patched_checkpoint

from nnterp.families import doge

REPO = "hf-tiny-v2/tiny-random-DogeForCausalLM"


def _gated_checkpoint(repo=REPO):
    """The tiny checkpoint with its residual gates set away from one.

    The tiny checkpoint's ``input_residual`` and ``post_attention_residual`` are
    their initial ones, where the gated identity is the plain one. The copy draws
    both from ``[0.5, 1.5)`` with a fixed seed and writes them into a copy of the
    weights, once per snapshot; everything else is symlinked.
    """
    from safetensors.torch import load_file, save_file

    snapshot = cached_snapshot(repo)
    patched = os.path.join(tempfile.gettempdir(), f"nnterp-doge-gated-{os.path.basename(snapshot)}")
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
    MID_IDENTITY = False                     # the residual gates scale the stream the block carries

    def test_gates_are_one_on_the_tiny_checkpoint(self, model):
        for layer in model.layers:
            assert torch.equal(layer._module.input_residual, torch.ones_like(layer._module.input_residual))
            assert torch.equal(layer._module.post_attention_residual, torch.ones_like(layer._module.post_attention_residual))

    def test_gated_identity(self, model):
        gated_identity(model)


class TestDogeGated(FamilySuite):
    """Gates away from one: the contributions are still the modules' outputs, and the identity carries the gates."""

    REPO = built(_gated_checkpoint)
    FAMILY = doge
    NATIVE = LLAMA_ROWS
    MID_IDENTITY = False                     # the residual gates scale the stream the block carries

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


def draw_gates(model):
    """The residual gates drawn from ``[0.5, 1.5)``: an initialization leaves them one, a trained Doge's are not."""
    for layer in model.model.layers:
        layer.input_residual.uniform_(0.5, 1.5)
        layer.post_attention_residual.uniform_(0.5, 1.5)


class TestDogeMoe(FamilySuite):
    """The cross-domain mixture (``is_moe``, as on Doge-40M-MoE): 16 product-key experts, top 4, behind gates away from
    one. It runs from transformers 5.18; its mixture values are unavailable either way. The tiny config with
    ``is_moe`` set, weights drawn for it, the gates too."""

    REPO = built(patched_checkpoint, REPO, init=draw_gates, is_moe=True, num_experts=16, num_experts_per_tok=4)
    FAMILY = doge
    NATIVE = LLAMA_ROWS
    MID_IDENTITY = False                     # the residual gates scale the stream the block carries
    ROWS_MIX = True                          # the mixture routes each token by product keys from other rows of the batch
    EXPECTED_UNAVAILABLE = {"mlp.mlp_activation": "", "mlp.mlp_neurons": ""}  # the mixture's interior: no one activation

    def test_contribution_identity(self, model):
        """The suite's identity with the block's gates on the residual."""
        gated_identity(model)

    def test_mixture_values_say_why(self, model):
        assert isinstance(model.layers[0].mlp, doge.Moe)
        support = model.support(layer=0)
        assert support["mlp.mlp_output"] is None
        assert all(support[f"mlp.{name}"] == doge.CANNOT_RUN for name in ("router_logits", "expert_weights", "routed_output"))

    @pytest.mark.skipif(Version(transformers.__version__) < Version("5.18"), reason="transformers < 5.18 cannot run DogeCDMoE")
    def test_mlp_output_is_the_mixtures_hidden_states(self, model):
        with model.trace(PROMPT):
            raw = model.layers[0].mlp.output[0].save()
            mlp = model.layers[0].mlp.mlp_output.save()
        assert torch.equal(mlp, raw)
        gated_identity(model)

    @pytest.mark.skipif(Version(transformers.__version__) >= Version("5.18"), reason="transformers >= 5.18 runs DogeCDMoE")
    def test_forward_fails_before_5_18(self, model):
        with pytest.raises(TypeError, match="dropout"):
            with model.trace(PROMPT):
                model.logits.save()
