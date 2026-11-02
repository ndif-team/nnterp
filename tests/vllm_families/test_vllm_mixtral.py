"""Mixtral on vLLM, against the transformers engine: a mixture of experts for an MLP, under its own name."""

from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import mixtral


class TestVLLMMixtral(VLLMFamilySuite):
    REPO = "TitanML/tiny-mixtral"
    FAMILY = mixtral
    UNAVAILABLE = VLLMFamilySuite.UNAVAILABLE | {"mlp.mlp_neurons"}
    NATIVE = {**LLAMA_ROWS, "layers.0.mlp": "model.layers.0.block_sparse_moe"}
