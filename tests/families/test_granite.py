"""Granite, end to end: scaled residual adds, scaled embeddings, logits divided after the head."""

import glob
import json
import os
import tempfile

import torch
from suite import FamilySuite, LLAMA_ROWS, PROMPT

from nnterp.families import granite

REPO = "hf-internal-testing/tiny-random-GraniteForCausalLM"


def _scaled_checkpoint(repo=REPO, **overrides):
    """The tiny checkpoint with its multipliers set away from 1.0, which is all the tiny config has.

    The weights and tokenizer are symlinked, only ``config.json`` is rewritten:
    ``residual_multiplier`` 0.22 (granite-4.1-3b's) makes the contributions differ from the modules'
    outputs, ``logits_scaling`` 4.0 makes the logits differ from the head's
    output, and ``embedding_multiplier`` 3.0 makes the first block's input
    differ from the embedding module's output. ``overrides`` rewrite further
    keys (a family's test turning one of its own options off).
    """
    snapshot = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*"))[0]
    patched = tempfile.mkdtemp(prefix="granite-scaled-")
    for name in os.listdir(snapshot):
        if name != "config.json":
            os.symlink(os.path.realpath(os.path.join(snapshot, name)), os.path.join(patched, name))
    config = json.load(open(os.path.join(snapshot, "config.json")))
    config.update(residual_multiplier=0.22, logits_scaling=4.0, embedding_multiplier=3.0, **overrides)
    json.dump(config, open(os.path.join(patched, "config.json"), "w"))
    return patched


class TestGranite(FamilySuite):
    REPO = REPO
    FAMILY = granite
    NATIVE = LLAMA_ROWS


class TestGraniteScaled(FamilySuite):
    """The same weights with every multiplier away from 1.0: the suite's identity then holds only on the scaled terms."""

    REPO = _scaled_checkpoint()
    FAMILY = granite
    NATIVE = LLAMA_ROWS

    def test_the_config_is_scaled(self, model):
        config = model.config
        assert (config.residual_multiplier, config.logits_scaling, config.embedding_multiplier) == (0.22, 4.0, 3.0)

    def test_contributions_are_the_scaled_module_outputs(self, model):
        with model.trace(PROMPT):
            attn_raw = model.layers[0].self_attn.output[0].save()
            attn = model.layers[0].self_attn.attention_output.save()
            mlp_raw = model.layers[0].mlp.output.save()
            mlp = model.layers[0].mlp.mlp_output.save()
        torch.testing.assert_close(attn, attn_raw * 0.22)
        torch.testing.assert_close(mlp, mlp_raw * 0.22)

    def test_in_place_edits_reach_the_model_through_the_transform(self, model):
        with model.trace(PROMPT):
            clean = model.logits.save()
        with model.trace(PROMPT):
            layer = model.layers[0]
            x = layer.input.save()
            attn = layer.self_attn.attention_output
            attn[:] = 0
            kept = attn.save()
            mlp = layer.mlp.mlp_output.save()
            out = layer.layer_output.save()
            edited = model.logits.save()
        assert not torch.equal(clean, edited)
        assert torch.equal(kept, torch.zeros_like(kept))  # the user's copy stays what they made it
        torch.testing.assert_close(out, x + mlp)  # the block added the zero
        with model.trace(PROMPT):
            layer = model.layers[0]
            x = layer.input.save()
            attn = layer.self_attn.attention_output.save()
            layer.mlp.mlp_output[:, -1] = 0
            out = layer.layer_output.save()
            edited = model.logits.save()
        assert not torch.equal(clean, edited)
        torch.testing.assert_close(out[:, -1], (x + attn)[:, -1])

    def test_assignment_reaches_the_model_scaled(self, model):
        """Assigning a contribution sets what the block adds, not the module's unscaled output."""
        with model.trace(PROMPT):
            layer = model.layers[0]
            x = layer.input.save()
            layer.self_attn.attention_output = torch.ones_like(x)
            mlp = layer.mlp.mlp_output.save()
            out = layer.layer_output.save()
        torch.testing.assert_close(out, x + 1 + mlp)

    def test_a_read_leaves_the_forward_untouched(self, model):
        with model.trace(PROMPT):
            clean = model.logits.save()
        with model.trace(PROMPT):
            for layer in model.layers:
                layer.self_attn.attention_output.save()
                layer.mlp.mlp_output.save()
            read = model.logits.save()
        assert torch.equal(clean, read)

    def test_token_embeddings_are_scaled_into_the_first_block(self, model):
        with model.trace(PROMPT):
            emb = model.token_embeddings.save()
            first = model.layers[0].input.save()
        torch.testing.assert_close(first, emb * 3.0)

    def test_logits_are_the_head_over_logits_scaling(self, model):
        with model.trace(PROMPT):
            resid = model.layers[-1].layer_output.save()
            raw = model.lm_head.output.save()
            logits = model.logits.save()
        torch.testing.assert_close(logits, raw / 4.0)
        torch.testing.assert_close(model.project_on_vocab(resid), logits)
