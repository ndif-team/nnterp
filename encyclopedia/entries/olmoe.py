"""OLMoE: Llama's pre-norm block with OLMo's query and key norms, and a mixture of 64 experts on every block."""

MODEL_TYPE = "olmoe"
TITLE = "OLMoE"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "allenai/OLMoE-1B-7B-0924"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-OlmoeForCausalLM"
CHECKPOINTS = [
    "allenai/OLMoE-1B-7B-0924", "allenai/OLMoE-1B-7B-0924-SFT", "allenai/OLMoE-1B-7B-0924-Instruct",
    "allenai/OLMoE-1B-7B-0125", "allenai/OLMoE-1B-7B-0125-SFT", "allenai/OLMoE-1B-7B-0125-DPO",
    "allenai/OLMoE-1B-7B-0125-Instruct",
]

#: Set by hues.py (lineage: OLMo).
PALETTE = {"hue": 104}
QUIRKS = ["qk-norm", "mixture-of-experts", "unnormalized-routing"]

BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} heads × {head_dim}, q/k normed",
        },
        {
            "host": "mlp",
            "kind": "moe",
            "label": "MoE",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "interior": [
                "router_logits", "expert_weights", "expert_indices",
                "expert_outputs", "routed_output",
            ],
            "detail": "{num_experts} experts × {intermediate_size}, top {top_k}",
            "pre_norm_note": "The mixture's input norm, named for its place after the attention's add.",
        },
    ],
}

STRIP: dict[str, str] = {}

NOTES = """
## `q_norm` and `k_norm` span every head

Each takes one RMS over the whole projection before the heads are split, so an edit to one head's
slice of `q_proj.output` or `k_proj.output` reaches the other heads. Edit one head at
`attention_queries` or `attention_keys`, which no norm follows.
"""
