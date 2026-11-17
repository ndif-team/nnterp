"""Seed-OSS on vLLM's transformers backend, against the transformers engine, on a tiny random checkpoint (the real one is 36B)."""

from vllm_suite import LLAMA_ROWS, NO_ROTARY, VLLMFamilySuite

from nnterp.families.vllm import seed_oss


class TestVLLMSeedOss(VLLMFamilySuite):
    REPO = "yujiepan/seed-oss-tiny-random"
    FAMILY = seed_oss
    UNAVAILABLE = VLLMFamilySuite.UNAVAILABLE | NO_ROTARY  # the transformers backend's rotary is not served
    NATIVE = LLAMA_ROWS
