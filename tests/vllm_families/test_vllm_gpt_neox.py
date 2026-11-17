"""GPT-NeoX (Pythia) on vLLM's transformers backend, against the transformers engine: transformers' tree under ``model``, a parallel block."""

from vllm_suite import NO_ROTARY, VLLMFamilySuite

from nnterp.families.vllm import gpt_neox


class TestVLLMGPTNeoX(VLLMFamilySuite):
    REPO = "EleutherAI/pythia-70m"
    FAMILY = gpt_neox
    UNAVAILABLE = (VLLMFamilySuite.UNAVAILABLE | NO_ROTARY | {"layer_mid"}) - {"mlp.mlp_activation"}
    NATIVE = {
        "embed_tokens": "model.embed_in",
        "layers": "model.layers",
        "norm": "model.final_layer_norm",
        "lm_head": "lm_head",
        "layers.0.self_attn": "model.layers.0.attention",
        "layers.0.mlp": "model.layers.0.mlp",
    }
    # Pythia-70m's queries reach the hundreds, so its softmax is as sharp as float32 allows and the two engines'
    # kernels part ways as depth compounds it: block 0's queries, keys and values agree to 5e-8 and its head outputs
    # to 1e-3, block 3's head outputs to 1.2e-2, and the last block's (queries near 930) to 1.2e-1.
    TOLERANCE = 6e-2
    KERNEL_TOLERANCE = 2e-1
