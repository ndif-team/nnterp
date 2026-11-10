"""Mixtral: Mistral's block with a mixture of 8 experts, 2 per token, in place of the MLP."""

MODEL_TYPE = "mixtral"
TITLE = "Mixtral"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "mistralai/Mixtral-8x7B-v0.1"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-MixtralForCausalLM"
CHECKPOINTS = [
    "mistralai/Mixtral-8x7B-v0.1", "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mixtral-8x22B-v0.1", "mistralai/Mixtral-8x22B-Instruct-v0.1",
]

#: Set by hues.py (lineage: Llama and its kin).
PALETTE = {"hue": 263}
QUIRKS = ["mixture-of-experts"]

#: Every number in the notes comes from a checkpoint's config or tokenizer; the shapes, identities and
#: snippets were run on the pinned tiny checkpoint (4 experts, top 2, 2 blocks), in float32 and bfloat16.
#: No real weights were run: the smallest checkpoint has 47B parameters.

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
            "detail": "{num_heads} heads, {num_kv_heads} kv",
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
        },
    ],
}

#: Notes on the model-level strip, by node: none, since nothing happens at the embeddings, the final norm or the head.
STRIP: dict[str, str] = {}

NOTES = ""
