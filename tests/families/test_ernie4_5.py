"""ERNIE 4.5 dense, end to end."""

from suite import FamilySuite, LLAMA_ROWS, built, patched_checkpoint

from nnterp.families import ernie4_5


class TestErnie45(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-Ernie4_5ForCausalLM"
    FAMILY = ernie4_5
    NATIVE = LLAMA_ROWS


class TestErnie45Wide(TestErnie45):
    """ERNIE-4.5-0.3B's layout: grouped queries (4 heads, 2 key/value heads here; 16 and 2 there) whose width (4 x 16
    here, 16 x 128 there) is not the stream's, and tied embeddings. Weights drawn for it."""

    REPO = built(patched_checkpoint, TestErnie45.REPO, draw=True, num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True)
