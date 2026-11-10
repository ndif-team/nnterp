"""GLM-4, end to end."""

from suite import FamilySuite, LLAMA_ROWS, built, patched_checkpoint

from nnterp.families import glm


class TestGlm(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-GlmForCausalLM"
    FAMILY = glm
    NATIVE = LLAMA_ROWS


class TestGlmEdge(TestGlm):
    """glm-edge's layout: the rotary over the whole head (``partial_rotary_factor`` 1.0, through GLM's interleaved
    partial function), no attention biases, grouped queries (one key/value head here; 4 of 16 there) and tied
    embeddings. Weights drawn for it."""

    REPO = built(
        patched_checkpoint, TestGlm.REPO, draw=True, partial_rotary_factor=1.0, rope_parameters={"partial_rotary_factor": 1.0},
        attention_bias=False, num_key_value_heads=1, tie_word_embeddings=True,
    )
