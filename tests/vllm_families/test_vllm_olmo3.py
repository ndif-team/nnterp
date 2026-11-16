"""OLMo 3 on vLLM's transformers backend, against the transformers engine: transformers' modules, post-norm."""

from vllm_suite import LLAMA_ROWS, NO_ROTARY, VLLMFamilySuite

from nnterp.families.vllm import olmo3


class TestVLLMOlmo3(VLLMFamilySuite):
    REPO = "allenai/Olmo-3-7B-Instruct"
    FAMILY = olmo3
    UNAVAILABLE = VLLMFamilySuite.UNAVAILABLE | NO_ROTARY | {"attention_norm_output", "mlp_norm_output"}
    NATIVE = LLAMA_ROWS
    MEMORY = 0.55
