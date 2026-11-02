"""GLM-4 (the first generation) on vLLM, against the transformers engine: vLLM's Llama classes."""

from vllm_suite import LLAMA_ROWS, VLLMFamilySuite

from nnterp.families.vllm import glm


class TestVLLMGlm(VLLMFamilySuite):
    REPO = "zai-org/glm-edge-1.5b-chat"
    FAMILY = glm
    NATIVE = LLAMA_ROWS
    MEMORY = 0.3
    # vLLM's GlmForCausalLM sets ``partial_rotary_factor`` to 0.5 whatever the checkpoint says, and GLM-Edge says
    # 1.0, so vLLM rotates half of each head where transformers rotates all of it (layer 0's queries differ by 44% of
    # their scale; the logits by 21%). The GLM checkpoints that do say 0.5 (GLM-4-9B) are 19 GB, so the reference is
    # GLM-Edge with the factor vLLM uses: the model vLLM runs, which is what this family maps.
    REFERENCE = {"rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5}}
