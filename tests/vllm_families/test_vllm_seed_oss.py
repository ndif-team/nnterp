"""Seed-OSS on vLLM, against the transformers engine, on a tiny random checkpoint (the real one is 36B)."""

from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import seed_oss


class TestVLLMSeedOss(VLLMFamilySuite):
    REPO = "yujiepan/seed-oss-tiny-random"
    FAMILY = seed_oss
    NATIVE = LLAMA_ROWS
