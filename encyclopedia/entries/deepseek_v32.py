"""DeepSeek-V3.2: V3's latent attention behind DeepSeek Sparse Attention, three dense first blocks, and a sigmoid-routed mixture."""

MODEL_TYPE = "deepseek_v32"
TITLE = "DeepSeek-V3.2"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "deepseek-ai/DeepSeek-V3.2"
#: The tiny checkpoint the test suite builds the page from (block 0 dense, block 1 a mixture; index_topk 2048).
PINNED = "hf-tiny-v2/tiny-random-DeepseekV32ForCausalLM"
#: Every public id whose config is deepseek_v32; the four share one set of sizes.
CHECKPOINTS = [
    "deepseek-ai/DeepSeek-V3.2", "deepseek-ai/DeepSeek-V3.2-Speciale",
    "deepseek-ai/DeepSeek-V3.2-Exp", "deepseek-ai/DeepSeek-V3.2-Exp-Base",
]

#: Set by hues.py (lineage: DeepSeek).
PALETTE = {"hue": 328}
QUIRKS = ["latent-attention", "sparse-attention", "interleaved-rotary", "mixture-of-experts", "dense-first-blocks"]

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

#: Notes on the model-level strip, by node: none, since nothing happens at the embeddings, the final norm or the head.
STRIP: dict[str, str] = {}

NOTES = """
## Every head shares the rotary part of the keys

Keys are expanded per head from one latent per token, but their last `qk_rope_head_dim` dimensions
come from one projection shared by every head: `attention_keys[:, h, :, -qk_rope_head_dim:]` is the
same tensor for every `h`. An edit to that part acts as an edit to the shared key only when it is
made on every head. `config.head_dim` is the rotary part's width, which no served value has.

## The score scale carries YaRN's factor

Every checkpoint extends its context with YaRN, and the attention multiplies the scores by
`qk_head_dim ** -0.5` times a factor set by `mscale_all_dim`. Recompute `attention_scores` from the
queries and keys with `self_attn.scaling`.

## The indexer's selection gets no gradient

The indexer runs under `torch.no_grad()`, so no gradient reaches its projections, and attribution
patching and other gradient methods treat the choice of keys as fixed.

## The expert weights sum to `routed_scaling_factor`

The router picks experts by the sigmoid of `router_logits` plus a selection bias,
`router.e_score_correction_bias`, and only from a token's best `topk_group` groups of experts, so
the top `router_logits` are not always the chosen experts. `expert_weights` are the chosen sigmoids
without the bias, renormalized and multiplied by `routed_scaling_factor`, so they sum to it at
every token.
"""
