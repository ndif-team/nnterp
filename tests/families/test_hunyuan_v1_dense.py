"""Hunyuan dense V1, end to end."""

from suite import FamilySuite, LLAMA_ROWS

from nnter.families import hunyuan_v1_dense


class TestHunyuanV1Dense(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-HunYuanDenseV1ForCausalLM"
    FAMILY = hunyuan_v1_dense
    NATIVE = LLAMA_ROWS
