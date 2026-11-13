"""GLM-4.7-Flash: DeepSeek-V2's latent attention on Llama's block, a dense first block and a sigmoid-routed mixture after."""

MODEL_TYPE = "glm4_moe_lite"
TITLE = "GLM-4.7-Flash"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "zai-org/GLM-4.7-Flash"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-Glm4MoeLiteForCausalLM"
#: The one public checkpoint whose config is glm4_moe_lite (GLM-4.7 itself is glm4_moe).
CHECKPOINTS = ["zai-org/GLM-4.7-Flash"]

#: Set by hues.py (lineage: GLM).
PALETTE = {"hue": 125}
QUIRKS = ["latent-attention", "interleaved-rotary", "mixture-of-experts", "dense-first-blocks"]

#: The sublayers in forward order; a block draws the MLP its ``mlp`` is (dense on block 0, a mixture after).
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
            "detail": "latent, {num_heads} heads, q·k {qk_head_dim}, v {v_head_dim}",
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

STRIP: dict[str, str] = {}

NOTES = """
## The heads share one rotary key, and `config.head_dim` is its width

The last `qk_rope_head_dim` dimensions of every head's key are one rotary key that all heads share.
The config's `head_dim` is that width (64), which no served value has. Queries and keys
are `model.qk_head_dim` wide and values `model.head_dim`.

## The expert weights sum to `routed_scaling_factor`

The router picks experts by the sigmoid of `router_logits` plus a selection bias,
`router.e_score_correction_bias`, so the top `router_logits` are not always the chosen experts.
`expert_weights` are the chosen sigmoids without the bias, renormalized and multiplied by
`routed_scaling_factor`, so a token's weights sum to 1.8. A shared expert runs on every token inside
`mlp_output`, which is `routed_output + shared_expert_output`.
"""
