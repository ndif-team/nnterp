"""Phi-3, end to end."""

import pytest
import torch
from suite import PROMPT, FamilySuite, LLAMA_ROWS, built, patched_checkpoint, random_weights

from nnterp.families import phi3


class TestPhi3(FamilySuite):
    REPO = "katuni4ka/tiny-random-phi3"
    FAMILY = phi3
    NATIVE = LLAMA_ROWS


class TestPhi3LongRope(TestPhi3):
    """Phi-3-mini-128k's, Phi-3.5-mini's and Phi-4-mini's layout: a ``longrope`` rotary, partial rotary 0.75,
    grouped-query attention (6 heads over 2 here, 24 over 8 on Phi-4-mini) and tied embeddings (Phi-4-mini). Its
    ``original_max_position_embeddings`` is 2 (4096 on the releases), so every prompt here takes the long factors,
    which the rotary computes on its first call and keeps. The tiny config with those set, weights drawn for it."""

    REPO = built(
        patched_checkpoint, TestPhi3.REPO, draw=True, num_attention_heads=6, num_key_value_heads=2, head_dim=8,
        partial_rotary_factor=0.75, original_max_position_embeddings=2, max_position_embeddings=256, tie_word_embeddings=True,
        rope_scaling=None, rope_parameters={
            "rope_type": "longrope", "rope_theta": 10000.0, "partial_rotary_factor": 0.75, "original_max_position_embeddings": 2,
            "short_factor": [1.0 + 0.1 * i for i in range(3)], "long_factor": [2.0 + 0.2 * i for i in range(3)],
        },
    )

    def test_the_rotary_takes_the_long_factors(self, model):
        rotary = model.model.rotary_emb._module
        assert rotary.rope_type == "longrope"
        if len(model.tokenizer(PROMPT)["input_ids"]) <= model.config.original_max_position_embeddings:
            pytest.skip("the prompt is within original_max_position_embeddings: the short factors apply")
        with model.trace(PROMPT):
            model.logits.save()
        assert type(rotary.long_inv_freq) is torch.Tensor and torch.equal(rotary.inv_freq, rotary.long_inv_freq)


def test_value_weight_under_grouped_query_attention():
    """``qkv_proj`` holds ``num_heads`` queries but ``num_key_value_heads`` keys and values; the tiny checkpoint has as
    many of each, so random weights."""
    model = random_weights(TestPhi3.REPO, num_attention_heads=4, num_key_value_heads=2)
    assert (model.num_heads, model.num_kv_heads) == (4, 2)
    FamilySuite().test_value_weight_reproduces_the_values(model)


#: The rope scaling of Phi-3-mini-128k, Phi-3.5-mini and Phi-4-mini, sized for the tiny checkpoint (head_dim 24).
LONGROPE = {
    "rope_type": "longrope", "rope_theta": 10000.0, "original_max_position_embeddings": 16,
    "short_factor": [1.0 + 0.1 * i for i in range(12)], "long_factor": [2.0 + 0.2 * i for i in range(12)],
}


def test_rank_before_the_first_trace_leaves_a_longrope_model_as_it_was():
    """A longrope rotary embedding computes its long frequencies on its first call and keeps them; the probe behind
    `rank` runs a dispatched model for real, so what it leaves there is a tensor, and the traces after it are the
    traces of a model never ranked."""
    def load():
        return random_weights(TestPhi3.REPO, rope_parameters=LONGROPE, original_max_position_embeddings=16)

    def logits(model):
        with torch.no_grad(), model.trace(PROMPT):
            out = model.logits.save()
        return out

    clean = logits(load())
    ranked = load()
    assert ranked.dispatched and ranked.rank("layer_output", 0) == (0, ranked.order(0)["layer_output"])
    rotary = ranked.model.rotary_emb._module
    assert type(rotary.long_inv_freq) is torch.Tensor and type(rotary.inv_freq) is torch.Tensor
    assert torch.equal(logits(ranked), clean)
