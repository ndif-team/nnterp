"""Phi-3, end to end."""

from suite import FamilySuite, LLAMA_ROWS, random_weights

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
