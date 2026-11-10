"""Seed-OSS, end to end."""

from suite import FamilySuite, LLAMA_ROWS, built, draw_biases, patched_checkpoint

from nnterp.families import seed_oss


class TestSeedOss(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-SeedOssForCausalLM"
    FAMILY = seed_oss
    NATIVE = LLAMA_ROWS


class TestSeedOssWide(TestSeedOss):
    """Seed-OSS-36B's attention: grouped queries (4 heads, 2 key/value heads here; 80 and 8 there) whose width
    (4 x 16 here, 80 x 128 there) is not the stream's, with biased projections. Weights drawn for it, the biases too."""

    REPO = built(patched_checkpoint, TestSeedOss.REPO, init=draw_biases, num_attention_heads=4, num_key_value_heads=2)
