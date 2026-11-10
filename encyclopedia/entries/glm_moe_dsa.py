"""GLM-5: latent attention with DeepSeek Sparse Attention, the selection carried between blocks, and a sigmoid-routed mixture."""

MODEL_TYPE = "glm_moe_dsa"
TITLE = "GLM-5"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "zai-org/GLM-5"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-GlmMoeDsaForCausalLM"
#: Every public id whose config is glm_moe_dsa. GLM-5.3-Flash is glm5_next, another family.
CHECKPOINTS = [
    "zai-org/GLM-5", "zai-org/GLM-5-FP8", "zai-org/GLM-5.1", "zai-org/GLM-5.1-FP8",
    "zai-org/GLM-5.2", "zai-org/GLM-5.2-FP8", "zai-org/GLM-5.3", "zai-org/GLM-5.3-BF16",
]

#: Set by hues.py (lineage: GLM).
PALETTE = {"hue": 128}
QUIRKS = ["latent-attention", "sparse-attention", "interleaved-rotary", "mixture-of-experts", "dense-first-blocks"]

#: Shapes, identities, read orders, the rotary order, the selection and the routing were run on the pinned tiny
#: checkpoint (block 0 a mixture of 8 experts, top 2, block 1 dense; queries, keys and values 128 wide;
#: index_topk 2048) and on the suite's copy with index_topk 2. Sizes, scales, the indexer layout, the routing
#: settings and the tokenizer are the Hub configs', meta builds and tokenizers; no real weights were run
#: (no checkpoint of the family is small enough).

#: The sublayers in forward order; a block draws the MLP its ``mlp`` is (dense on blocks 0-2, a mixture after).
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
            "detail": "sparse latent, {num_heads} heads, q·k {qk_head_dim}, v {v_head_dim}",
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

The router picks the top 8 experts by sigmoid score plus `router.e_score_correction_bias`. The bias
decides the choice only. `expert_weights` are the chosen sigmoids without it, renormalized and
multiplied by `routed_scaling_factor`, so a token's weights sum to 2.5. A shared expert runs on every
token inside `mlp_output`, which is `routed_output + shared_expert_output`.

## The tokenizer adds no beginning-of-sequence token

The sink that trained models form at position 0 falls on the prompt's first token. Leave position 0
out of activation statistics and steering-vector norms.
"""
