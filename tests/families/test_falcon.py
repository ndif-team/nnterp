"""Falcon (7B layout), end to end: parallel block, multi-query, in-place add into the MLP output."""

import glob
import json
import os
import tempfile

import nnsight
import torch
from suite import FamilySuite, rows, PROMPT

from nnter.families import falcon


class TestFalcon(FamilySuite):
    REPO = "Rocketknight1/tiny-random-falcon-7b"
    FAMILY = falcon
    NATIVE = rows("transformer", "h", "word_embeddings", "ln_f", attn="self_attention", ln2=None)
    MLP_NORM = "input_layernorm"             # parallel (7B layout)

    def test_mlp_output_is_a_copy_the_block_does_not_touch(self, model):
        """The block adds the attention into the MLP's live tensor in place; the value is a copy."""
        with model.trace(PROMPT):
            attn = model.layers[0].self_attn.attention_output.save()
            mlp = model.layers[0].mlp.mlp_output.save()
            live = model.layers[0].mlp.output.save()
        torch.testing.assert_close(live, mlp + attn)

    def test_in_place_mlp_edit_reaches_the_model_through_the_transform(self, model):
        with model.trace(PROMPT):
            clean = model.logits.save()
        with model.trace(PROMPT):
            model.layers[0].mlp.mlp_output[:] = 0
            kept = model.layers[0].mlp.mlp_output.save()
            edited = model.logits.save()
        assert not torch.equal(clean, edited)
        assert torch.equal(kept, torch.zeros_like(kept))  # the user's copy stays what they made it

    def test_a_statement_that_reads_the_copy_twice_edits_one_tensor(self, model):
        """``x.value[...] += f(x.value)`` reads twice; both reads are the copy the write-back carries."""
        mlp = model.layers[0].mlp
        with model.trace(PROMPT):
            first = mlp.mlp_output
            same = nnsight.save(mlp.mlp_output is first)
            first[:] += 1
            once = model.logits.save()
        with model.trace(PROMPT):
            mlp.mlp_output[:] += torch.ones_like(mlp.mlp_output)
            twice = model.logits.save()
        assert same and torch.equal(once, twice)

    def test_the_copy_is_read_anew_on_every_generation_step(self, model):
        """One copied value a step and nothing between: each step's is the model's next call, not the last copy."""
        mlp = model.layers[0].mlp
        with model.generate(PROMPT, max_new_tokens=3, min_new_tokens=3, do_sample=False) as tracer:
            copies = nnsight.save([])
            for step in tracer.iter[:3]:
                copies.append(mlp.mlp_output)
        assert [c.shape[1] for c in copies[1:]] == [1, 1] and copies[0].shape[1] > 1
        assert not torch.equal(copies[1], copies[2])

    def test_values_bind_before_the_rotary(self, model):
        """In one trace the values must be read before the queries or keys."""
        with model.trace(PROMPT):
            v = model.layers[0].self_attn.attention_values.save()
            q = model.layers[0].self_attn.attention_queries.save()
        assert v.shape[1] == 1 and q.shape[1] == model.num_heads  # multi-query



def _alibi_checkpoint(repo="Rocketknight1/tiny-random-falcon-7b"):
    """The 7B tiny checkpoint with ``alibi`` switched on in its config: the same weights, the other attention branch."""
    snapshot = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*"))[0]
    patched = tempfile.mkdtemp(prefix="falcon-alibi-")
    for name in os.listdir(snapshot):
        if name != "config.json":
            os.symlink(os.path.realpath(os.path.join(snapshot, name)), os.path.join(patched, name))
    config = json.load(open(os.path.join(snapshot, "config.json")))
    config["alibi"] = True
    json.dump(config, open(os.path.join(patched, "config.json"), "w"))
    return patched


class TestFalconAlibi(FamilySuite):
    """The 7B layout with alibi: no rotary, the pattern at the dropout after the second softmax, flattened head outputs."""

    REPO = _alibi_checkpoint()
    FAMILY = falcon
    NATIVE = rows("transformer", "h", "word_embeddings", "ln_f", attn="self_attention", ln2=None)
    MLP_NORM = "input_layernorm"

    def test_alibi_branch(self, model):
        assert model.layers[0].self_attn._module.config.alibi
        with model.trace(PROMPT):
            probs = model.layers[0].self_attn.attention_probabilities.save()
            raw = model.layers[0].self_attn.source.self_attention_dropout_0.output.save()
        assert torch.equal(probs, raw)


class TestFalcon40B(FamilySuite):
    """The 40B layout: ``new_decoder_architecture``, with ``ln_attn`` / ``ln_mlp`` in place of one norm."""

    REPO = "Rocketknight1/tiny-random-falcon-40b"
    FAMILY = falcon
    NATIVE = rows("transformer", "h", "word_embeddings", "ln_f", attn="self_attention", ln1=None, ln2=None)
    KV_HEADS_EXPANDED = True  # the new layout broadcasts its 8 kv heads to all 128 before the rotary
    ATTENTION_NORM = "ln_attn"
    MLP_NORM = "ln_mlp"
    MLP_NORM_BEFORE_ATTENTION = True   # both norms are taken from the block input before either sublayer runs

    def test_new_decoder_architecture(self, model):
        assert model.config.new_decoder_architecture
        assert hasattr(model.layers[0], "ln_attn") and hasattr(model.layers[0], "ln_mlp")
