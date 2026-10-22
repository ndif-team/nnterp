"""EXAONE 4.0 on vLLM, against the transformers engine: a post-norm block that returns a pair."""

from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import exaone4


class TestVLLMExaone4(VLLMFamilySuite):
    REPO = "LGAI-EXAONE/EXAONE-4.0-1.2B"
    FAMILY = exaone4
    UNAVAILABLE = VLLMFamilySuite.UNAVAILABLE | {"attention_norm_output", "mlp_norm_output"}
    NATIVE = LLAMA_ROWS
