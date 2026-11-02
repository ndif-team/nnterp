"""Hunyuan dense V1 on vLLM, against the transformers engine: a fused block that also returns the keys and values."""

from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import hunyuan_v1_dense


class TestVLLMHunYuanDenseV1(VLLMFamilySuite):
    REPO = "tencent/Hunyuan-0.5B-Instruct"
    FAMILY = hunyuan_v1_dense
    NATIVE = LLAMA_ROWS
    MEMORY = 0.3
