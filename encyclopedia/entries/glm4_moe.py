"""GLM-4.5, GLM-4.5-Air, GLM-4.6 and GLM-4.7: ``Glm4MoeForCausalLM``, dense first blocks and a sigmoid-routed mixture after."""

MODEL_TYPE = "glm4_moe"
TITLE = "GLM-4.5 / GLM-4.6 / GLM-4.7"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "zai-org/GLM-4.5-Air"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "trl-internal-testing/tiny-Glm4MoeForCausalLM"
CHECKPOINTS = [
    "zai-org/GLM-4.5-Air-Base", "zai-org/GLM-4.5-Air",
    "zai-org/GLM-4.5-Base", "zai-org/GLM-4.5",
    "zai-org/GLM-4.6", "zai-org/GLM-4.7",
]

#: Set by hues.py (lineage: GLM).
PALETTE = {"hue": 122}
QUIRKS = [
    "mixture-of-experts",
    "dense-first-blocks",
    "qkv-bias",
    {"slug": "qk-norm", "when": lambda config: config.use_qk_norm},  # GLM-4.5, 4.6, 4.7; not Air
    "partial-rotary",
]

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
            "detail": "{num_heads} heads, {num_kv_heads} kv, × {head_dim}",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
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
## The expert weights sum to `routed_scaling_factor`

The router picks experts by the sigmoid of `router_logits` plus a selection bias,
`router.e_score_correction_bias`, so the top `router_logits` are not always the chosen experts.
`expert_weights` are the chosen sigmoids without the bias, renormalized and multiplied by
`routed_scaling_factor`, so a token's weights sum to 2.5 on GLM-4.5, 4.6 and 4.7 and to 1.0 on Air. A
shared expert runs on every token inside `mlp_output`, which is `routed_output + shared_expert_output`.

## On GLM-4.7 the chat template ends where a reasoning trace starts

With `add_generation_prompt=True` the template ends the prompt with `<think>`, so a next-token metric
there measures the start of a reasoning trace, not the answer.
"""
