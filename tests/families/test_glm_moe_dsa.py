"""GLM-5, end to end: latent attention with DeepSeek Sparse Attention, the selection shared across blocks."""

import glob
import json
import os
import tempfile

import pytest
import torch
from suite import FamilySuite, LLAMA_ROWS, PROMPT

from nnterp import StandardizedTransformer
from nnterp.families import glm_moe_dsa

REPO = "hf-tiny-v2/tiny-random-GlmMoeDsaForCausalLM"


def _patched_checkpoint(repo=REPO, **changes):
    """The tiny checkpoint with ``changes`` written into its config: the same weights, symlinked."""
    snapshot = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*"))[0]
    patched = tempfile.mkdtemp(prefix="glm-moe-dsa-")
    for name in os.listdir(snapshot):
        if name != "config.json":
            os.symlink(os.path.realpath(os.path.join(snapshot, name)), os.path.join(patched, name))
    config = json.load(open(os.path.join(snapshot, "config.json")))
    config.update(changes)
    json.dump(config, open(os.path.join(patched, "config.json"), "w"))
    return patched


def selection_mask(indices, keys):
    """``[batch, query, topk]`` key indices -> ``[batch, 1, query, key]`` bool, True where selected and causal."""
    selected = torch.zeros(*indices.shape[:2], keys, dtype=torch.bool, device=indices.device)
    selected.scatter_(-1, indices.long(), True)
    return (selected & torch.ones(keys, keys, dtype=torch.bool, device=indices.device).tril()).unsqueeze(1)


def read_selection(model, layer):
    """One block's scores, pattern and selection; the selection is the attention's third return on every block."""
    with model.trace(PROMPT):
        scores = layer.self_attn.attention_scores.save()
        probs = layer.self_attn.attention_probabilities.save()
        indices = layer.self_attn.output[2].save()
    return scores, probs, indices


def assert_pattern_is_zero_outside_the_selection(model, sparse):
    for layer in model.layers:
        scores, probs, indices = read_selection(model, layer)
        mask = selection_mask(indices, probs.shape[-1]).expand_as(probs)
        assert torch.equal(probs != 0, mask)
        assert (scores[~mask] == torch.finfo(scores.dtype).min).all()
        assert mask[..., -1, :].all() != sparse  # the last row drops a key only under a small index_topk


def assert_written_pattern_moves_the_logits_on_every_block(model):
    """On every block, a pattern that puts the last query's weight on a key the selection dropped (or, dense, on the first key) moves the logits."""
    with model.trace(PROMPT):
        clean = model.logits.save()
    for layer in model.layers:
        _, probs, indices = read_selection(model, layer)
        dropped = [k for k in range(probs.shape[-1]) if k not in indices[0, -1].tolist()] or [0]
        written = probs.clone()
        written[..., -1, :] = 0
        written[..., -1, dropped[0]] = 1
        with model.trace(PROMPT):
            layer.self_attn.attention_probabilities = written
            moved = model.logits.save()
        assert not torch.allclose(clean, moved), layer


class TestGlmMoeDsa(FamilySuite):
    """The pinned checkpoint: ``index_topk`` is 2048, so on the suite's prompt every causal key is selected."""

    REPO = REPO
    FAMILY = glm_moe_dsa
    NATIVE = LLAMA_ROWS
    KV_HEADS_EXPANDED = True  # latent attention projects keys and values for every head

    def test_latent_attention_widths(self, model):
        assert model.qk_head_dim == model.config.qk_nope_head_dim + model.config.qk_rope_head_dim
        assert model.head_dim == model.config.v_head_dim

    def test_block_returns_the_selection_beside_the_stream(self, model):
        with model.trace(PROMPT):
            selection = model.layers[0].self_attn.output[2].save()
            out = model.layers[0].output.save()
            stream = model.layers[0].layer_output.save()
        assert isinstance(out, tuple) and len(out) == 2
        assert stream is out[0] and torch.equal(out[1], selection)

    def test_a_prompt_shorter_than_topk_selects_every_causal_key(self, model):
        assert_pattern_is_zero_outside_the_selection(model, sparse=False)

    def test_written_pattern_moves_the_logits_on_every_block(self, model):
        assert_written_pattern_moves_the_logits_on_every_block(model)


class TestGlmMoeDsaSparse(FamilySuite):
    """The same weights with ``index_topk = 2``: the suite's three-token prompt then drops a key from the last row."""

    REPO = _patched_checkpoint(index_topk=2)
    FAMILY = glm_moe_dsa
    NATIVE = LLAMA_ROWS
    KV_HEADS_EXPANDED = True

    def test_pattern_is_zero_outside_the_selection(self, model):
        assert_pattern_is_zero_outside_the_selection(model, sparse=True)

    def test_written_pattern_moves_the_logits_on_every_block(self, model):
        """The written pattern is used as written: weight on a key the indexer dropped reaches the logits."""
        assert_written_pattern_moves_the_logits_on_every_block(model)


class TestGlmMoeDsaShared:
    """Block 1 marked ``"shared"``: it has no indexer and reuses block 0's selection.

    Not a `FamilySuite`: the suite skips block 0 alone, which hands a shared block no
    selection. The unexpected indexer weights of block 1 are dropped at load.
    """

    @pytest.fixture(scope="class")
    def model(self):
        return StandardizedTransformer(_patched_checkpoint(index_topk=2, indexer_types=["full", "shared"]), dispatch=True, attn_implementation="eager")

    def test_shared_block_reuses_the_previous_selection(self, model):
        assert model.layers[1].self_attn._module.indexer is None
        with model.trace(PROMPT):
            first = model.layers[0].self_attn.indexer.output.save()
            second = model.layers[1].self_attn.output[2].save()
        assert torch.equal(first, second)

    def test_pattern_is_zero_outside_the_selection(self, model):
        assert_pattern_is_zero_outside_the_selection(model, sparse=True)

    def test_written_pattern_moves_the_logits_on_every_block(self, model):
        assert_written_pattern_moves_the_logits_on_every_block(model)

    def test_contribution_identity(self, model):
        for layer in model.layers:
            with model.trace(PROMPT):
                inp = layer.input.save()
                attn = layer.self_attn.attention_output.save()
                mlp = layer.mlp.mlp_output.save()
                out = layer.layer_output.save()
            torch.testing.assert_close(inp + attn + mlp, out)

    def test_skipping_into_a_shared_block_is_refused(self, model):
        """A skipped full block selects nothing, and the shared block after it has no selection to reuse."""
        with pytest.raises(Exception, match="Shared DSA layers require top-k indices"):
            with model.trace(PROMPT):
                model.skip_layers(0, 0)
                model.logits.save()
