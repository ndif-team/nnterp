"""OLMo 2 on vLLM's transformers backend, against the transformers engine: transformers' modules, post-norm."""

from vllm_suite import LLAMA_ROWS, NO_ROTARY, VLLMFamilySuite

from nnterp.families.vllm import olmo2


class TestVLLMOlmo2(VLLMFamilySuite):
    REPO = "allenai/OLMo-2-0425-1B"
    FAMILY = olmo2
    UNAVAILABLE = VLLMFamilySuite.UNAVAILABLE | NO_ROTARY | {"attention_norm_output", "mlp_norm_output"}
    NATIVE = LLAMA_ROWS
    MEMORY = 0.3
