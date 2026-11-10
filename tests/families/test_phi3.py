"""Phi-3, end to end."""

import torch
from suite import PROMPT, FamilySuite, LLAMA_ROWS, random_weights

from nnterp.families import phi3


class TestPhi3(FamilySuite):
    REPO = "katuni4ka/tiny-random-phi3"
    FAMILY = phi3
    NATIVE = LLAMA_ROWS


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
