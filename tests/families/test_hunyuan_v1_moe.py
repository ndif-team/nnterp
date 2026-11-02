"""Hunyuan MoE V1, end to end."""

from suite import FamilySuite, LLAMA_ROWS

from nnterp.families import hunyuan_v1_moe


class TestHunyuanV1Moe(FamilySuite):
    REPO = "hf-tiny-v2/tiny-random-HunYuanMoEV1ForCausalLM"
    FAMILY = hunyuan_v1_moe
    NATIVE = LLAMA_ROWS
