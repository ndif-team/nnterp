"""MiniMax-M2 on vLLM, against the transformers engine: whole-projection query and key norms, a mixture of experts under its own name."""

from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import minimax_m2


class TestVLLMMinimaxM2(VLLMFamilySuite):
    REPO = "yujiepan/minimax-m2-tiny-random"
    FAMILY = minimax_m2
    NATIVE = {**LLAMA_ROWS, "layers.0.mlp": "model.layers.0.block_sparse_moe"}
