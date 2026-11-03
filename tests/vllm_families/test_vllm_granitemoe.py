"""GraniteMoE on vLLM, against the transformers engine: Granite's multipliers around a mixture of experts."""

from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import granitemoe


class TestVLLMGraniteMoe(VLLMFamilySuite):
    REPO = "ibm-granite/granite-3.1-1b-a400m-instruct"
    FAMILY = granitemoe
    UNAVAILABLE = VLLMFamilySuite.UNAVAILABLE | {"mlp.mlp_neurons"}
    NATIVE = {**LLAMA_ROWS, "layers.0.mlp": "model.layers.0.block_sparse_moe"}
    MEMORY = 0.3
    # Position 0's key written at the last position before block 12's rotary: block 12's stream there still agrees
    # with transformers' to 4e-4 of its scale, and the next block's, after its mixture of experts, jumps to 1.1e-2
    # (3.5e-4 without the write); the logits differ by 6.4e-3. The same write of a query moves them by 7e-4.
    KERNEL_TOLERANCE = 1e-2
