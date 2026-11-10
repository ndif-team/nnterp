"""dots.llm1: Llama's pre-norm block with per-head query and key norms, one dense first block, then DeepSeek-V3's
sigmoid-routed mixture with a selection bias and two shared experts."""

MODEL_TYPE = "dots1"
TITLE = "dots.llm1"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "rednote-hilab/dots.llm1.base"
#: The tiny checkpoint the test suite builds the page from (every block a mixture of 8 experts, all 8 per token).
PINNED = "hf-tiny-v2/tiny-random-Dots1ForCausalLM"
CHECKPOINTS = ["rednote-hilab/dots.llm1.base", "rednote-hilab/dots.llm1.inst"]

#: Set by hues.py (no kin, in a gap between lineages).
PALETTE = {"hue": 317}
QUIRKS = [
    "qk-norm", "mixture-of-experts",
    # first_k_dense_replace is 1 on both checkpoints, 0 on the pinned tiny checkpoint
    {"slug": "dense-first-blocks", "when": lambda config: config.first_k_dense_replace > 0},
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
            "detail": "{num_heads} heads × {head_dim}, q/k normed",
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

#: Notes on the model-level strip, by node: none, since nothing happens at the embeddings, the final norm or the head.
STRIP: dict[str, str] = {}

NOTES = """
## The expert weights sum to `routed_scaling_factor`

The router picks experts by the sigmoid of `router_logits` plus a selection bias,
`router.e_score_correction_bias`, so the top `router_logits` are not always the chosen experts.
`expert_weights` are the chosen sigmoids without the bias, renormalized and multiplied by
`routed_scaling_factor`, 2.5, so they sum to 2.5 at every token.
"""
