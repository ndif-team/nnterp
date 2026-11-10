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

#: Every claim in the notes was checked on the pinned tiny checkpoint (4 experts, top 2, block 0 dense),
#: on a copy of its config with head_dim 8 for the rotary, or read off the listed checkpoints' configs and
#: tokenizers; no real weights were run.

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

The router picks the top 8 experts by sigmoid score plus `router.e_score_correction_bias`. The bias
decides the choice only. `expert_weights` are the chosen sigmoids without it, renormalized and
multiplied by `routed_scaling_factor`, so a token's weights sum to 2.5 on GLM-4.5, 4.6 and 4.7 and
to 1.0 on Air. A shared expert runs on every token inside `mlp_output`, which is
`routed_output + shared_expert_output`.

## The tokenizer prepends nothing

A raw prompt starts at position 0 with its first token, without the `[gMASK]<sop>` the chat template
writes, so the sink that trained models form at position 0 falls on that token. On GLM-4.7 the
generation prompt ends in `<think>`, so the next token starts a reasoning trace, not the answer.
"""
