"""Arcee, end to end."""

from suite import FamilySuite, LLAMA_ROWS, built, patched_checkpoint

from nnterp.families import arcee


class TestArcee(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-ArceeForCausalLM"
    FAMILY = arcee
    NATIVE = LLAMA_ROWS
    NEURONS_FROM_ACTIVATION = False          # relu squared: down_proj takes act(up)**2


class TestArceeReleased(TestArcee):
    """AFM-4.5B's layout: ``relu2``, YaRN (factor 20; ``original_max_position_embeddings`` 64 here, 4096 there) and
    grouped-query attention (4 heads of 8, 2 key/value heads here; 20 and 4 there). Weights drawn for it."""

    REPO = built(
        patched_checkpoint, TestArcee.REPO, draw=True, hidden_act="relu2", num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        rope_parameters={"rope_type": "yarn", "factor": 20.0, "original_max_position_embeddings": 64},
    )
