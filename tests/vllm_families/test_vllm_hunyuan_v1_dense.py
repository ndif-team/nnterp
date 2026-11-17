"""Hunyuan dense V1 on vLLM's transformers backend, against the transformers engine: transformers' modules, Llama's block."""

from vllm_suite import LLAMA_ROWS, NO_ROTARY, VLLMFamilySuite

from nnterp.families.vllm import hunyuan_v1_dense


class TestVLLMHunYuanDenseV1(VLLMFamilySuite):
    REPO = "tencent/Hunyuan-0.5B-Instruct"
    FAMILY = hunyuan_v1_dense
    UNAVAILABLE = VLLMFamilySuite.UNAVAILABLE | NO_ROTARY  # the transformers backend's rotary is not served
    NATIVE = LLAMA_ROWS
    MEMORY = 0.3
