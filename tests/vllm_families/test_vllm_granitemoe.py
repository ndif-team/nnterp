"""GraniteMoE on vLLM, against the transformers engine: Granite's multipliers around a mixture of experts."""

from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import granitemoe


class TestVLLMGraniteMoe(VLLMFamilySuite):
    REPO = "ibm-granite/granite-3.1-1b-a400m-instruct"
    FAMILY = granitemoe
    NATIVE = {**LLAMA_ROWS, "layers.0.mlp": "model.layers.0.block_sparse_moe"}
    MEMORY = 0.3
