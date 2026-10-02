"""GPT-NeoX (Pythia) on vLLM, against the transformers engine: positions first, a plain block."""

from vllm_suite import VLLMFamilySuite

from nnterp.families.vllm import gpt_neox


class TestVLLMGPTNeoX(VLLMFamilySuite):
    REPO = "EleutherAI/pythia-70m"
    FAMILY = gpt_neox
    NATIVE = {
        "embed_tokens": "gpt_neox.embed_in",
        "layers": "gpt_neox.layers",
        "norm": "gpt_neox.final_layer_norm",
        "lm_head": "embed_out",
        "layers.0.self_attn": "gpt_neox.layers.0.attention",
        "layers.0.mlp": "gpt_neox.layers.0.mlp",
    }
    # Pythia-70m's queries reach the hundreds, so its softmax is as sharp as float32 allows and the two engines'
    # kernels part ways as depth compounds it: block 0's queries, keys and values agree to 5e-8 and its head outputs
    # to 1e-3, block 3's head outputs to 1.2e-2, and the last block's (queries near 930) to 1.2e-1.
    TOLERANCE = 6e-2
    KERNEL_TOLERANCE = 2e-1
