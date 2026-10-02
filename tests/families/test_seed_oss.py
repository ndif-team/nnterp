"""Seed-OSS, end to end."""

from suite import FamilySuite, LLAMA_ROWS

from nnter.families import seed_oss


class TestSeedOss(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-SeedOssForCausalLM"
    FAMILY = seed_oss
    NATIVE = LLAMA_ROWS
