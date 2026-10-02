"""DeepSeek-V3.2, end to end: latent attention with DeepSeek Sparse Attention."""

import glob
import json
import os
import tempfile

import torch
from suite import FamilySuite, LLAMA_ROWS, PROMPT

from nnter.families import deepseek_v32

REPO = "hf-tiny-v2/tiny-random-DeepseekV32ForCausalLM"


def _patched_checkpoint(repo=REPO, **changes):
    """The tiny checkpoint with ``changes`` written into its config: the same weights, symlinked."""
    snapshot = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*"))[0]
    patched = tempfile.mkdtemp(prefix="deepseek-v32-")
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


def assert_written_pattern_moves_the_logits_on_every_block(model):
    """On every block, a pattern that puts the last query's weight on a key the indexer dropped (or, dense, on the first key) moves the logits."""
    with model.trace(PROMPT):
        clean = model.logits.save()
    for layer in model.layers:
        with model.trace(PROMPT):
            indices = layer.self_attn.indexer.output.save()
            probs = layer.self_attn.attention_probabilities.save()
        keys = probs.shape[-1]
        dropped = [k for k in range(keys) if k not in indices[0, -1].tolist()] or [0]
        written = probs.clone()
        written[..., -1, :] = 0
        written[..., -1, dropped[0]] = 1
        with model.trace(PROMPT):
            layer.self_attn.attention_probabilities = written
            moved = model.logits.save()
        assert not torch.allclose(clean, moved), layer


class TestDeepseekV32(FamilySuite):
    """The pinned checkpoint: ``index_topk`` is 2048, so on the suite's prompt every causal key is selected."""

    REPO = REPO
    FAMILY = deepseek_v32
    NATIVE = LLAMA_ROWS
    KV_HEADS_EXPANDED = True  # latent attention projects keys and values for every head

    def test_latent_attention_widths(self, model):
        assert model.qk_head_dim == model.config.qk_nope_head_dim + model.config.qk_rope_head_dim
        assert model.head_dim == model.config.v_head_dim

    def test_a_prompt_shorter_than_topk_selects_every_causal_key(self, model):
        with model.trace(PROMPT):
            indices = model.layers[0].self_attn.indexer.output.save()
            probs = model.layers[0].self_attn.attention_probabilities.save()
        keys = probs.shape[-1]
        assert keys < model.config.index_topk and indices.shape[-1] == keys
        assert torch.equal(probs != 0, selection_mask(indices, keys).expand_as(probs))
        assert (probs[0, 0].tril() != 0).sum() == keys * (keys + 1) // 2  # dense causal

    def test_written_pattern_moves_the_logits_on_every_block(self, model):
        assert_written_pattern_moves_the_logits_on_every_block(model)


class TestDeepseekV32Sparse(FamilySuite):
    """The same weights with ``index_topk = 2``: the suite's three-token prompt then drops a key from the last row."""

    REPO = _patched_checkpoint(index_topk=2)
    FAMILY = deepseek_v32
    NATIVE = LLAMA_ROWS
    KV_HEADS_EXPANDED = True

    def test_pattern_is_zero_outside_the_selection(self, model):
        for layer in model.layers:
            with model.trace(PROMPT):
                indices = layer.self_attn.indexer.output.save()
                scores = layer.self_attn.attention_scores.save()
                probs = layer.self_attn.attention_probabilities.save()
            keys = probs.shape[-1]
            mask = selection_mask(indices, keys).expand_as(probs)
            assert indices.shape[-1] == 2 and not mask[..., -1, :].all()  # the last row is sparse
            assert torch.equal(probs != 0, mask)
            assert (scores[~mask] == torch.finfo(scores.dtype).min).all()

    def test_written_pattern_moves_the_logits_on_every_block(self, model):
        """The written pattern is used as written: weight on a key the indexer dropped reaches the logits."""
        assert_written_pattern_moves_the_logits_on_every_block(model)
