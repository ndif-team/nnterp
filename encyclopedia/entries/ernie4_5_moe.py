"""ERNIE 4.5 MoE: ERNIE 4.5's block with dense first blocks, then a softmax-routed mixture whose choice adds a
correction bias to the probabilities, and on 21B-A3B a shared expert that runs before the router."""

MODEL_TYPE = "ernie4_5_moe"
TITLE = "ERNIE 4.5 MoE"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "baidu/ERNIE-4.5-21B-A3B-Base-PT"
#: The tiny checkpoint the test suite builds the page from (both blocks a mixture of 8 experts, top 2, 2 shared).
PINNED = "hf-tiny-v2/tiny-random-Ernie4_5_MoeForCausalLM"
#: The 300B-A47B checkpoints' mixture has no shared expert: their page draws no shared panel.
CHECKPOINTS = [
    "baidu/ERNIE-4.5-21B-A3B-Base-PT", "baidu/ERNIE-4.5-21B-A3B-PT", "baidu/ERNIE-4.5-21B-A3B-Thinking",
    "baidu/ERNIE-4.5-300B-A47B-Base-PT", "baidu/ERNIE-4.5-300B-A47B-PT",
]

#: Set by hues.py (lineage: ERNIE).
PALETTE = {"hue": 189}
QUIRKS = [
    "interleaved-rotary", "mixture-of-experts",
    # moe_layer_start_index is 1 on every checkpoint, 0 on the pinned tiny checkpoint
    {"slug": "dense-first-blocks", "when": lambda config: config.moe_layer_start_index > 0},
]

#: The smallest checkpoint has 21B parameters: nothing here ran on real weights. Every identity, shape, read
#: order and snippet ran on the pinned tiny checkpoint, loaded with moe_layer_start_index 1 for the dense-block
#: claims and with a correction bias written by hand for the bias claims. Sizes and block layouts are the Hub
#: configs', read on meta builds.

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
            "detail": "{num_heads} heads × {head_dim}, {num_kv_heads} kv",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
        {
            "host": "mlp",
            "kind": "moe",
            "part_order": ["shared", "router", "experts"],
            "label": "MoE",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "interior": [
                "router_logits", "expert_weights", "expert_indices",
                "expert_outputs", "routed_output", "shared_expert_output",
            ],
            "detail": "{num_experts} experts × {moe_intermediate_size}, top {top_k}",
        },
    ],
}

#: Notes on the model-level strip, by node: none, since nothing happens at the embeddings, the final norm or the head.
STRIP: dict[str, str] = {}

NOTES = """
## The router's selection bias is added to probabilities

The router takes the softmax of `router_logits` and chooses experts by those probabilities plus a
correction bias, `router.moe_statics.e_score_correction_bias`. The bias outweighs any smaller gap
between two probabilities, so the top `router_logits` are not always the chosen experts.
`expert_weights` are the chosen probabilities without the bias, renormalized to sum to one.
"""
