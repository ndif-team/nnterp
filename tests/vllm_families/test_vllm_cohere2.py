"""Cohere2 on vLLM, against the transformers engine: a parallel block that returns a pair, no ``lm_head`` module."""

from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import cohere2


class TestVLLMCohere2(VLLMFamilySuite):
    REPO = "israel/AfriGuard-tiny-aya-global-full"
    FAMILY = cohere2
    NATIVE = {name: path for name, path in LLAMA_ROWS.items() if name != "lm_head"}   # vLLM unembeds with embed_tokens
    MEMORY = 0.4
