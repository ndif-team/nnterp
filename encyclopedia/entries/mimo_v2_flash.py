"""MiMo-V2-Flash: sliding-window blocks with a sink and twice the key/value heads among full blocks, queries and keys
wider than values, and a sigmoid-routed mixture after a dense first block."""

MODEL_TYPE = "mimo_v2_flash"
TITLE = "MiMo-V2-Flash"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "XiaomiMiMo/MiMo-V2-Flash"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-MiMoV2FlashForCausalLM"
#: MiMo-V2.5 and V2.6 have model_type mimo_v2, another architecture, and are not listed.
CHECKPOINTS = ["XiaomiMiMo/MiMo-V2-Flash", "XiaomiMiMo/MiMo-V2-Flash-Base"]

#: Set by hues.py (lineage: MiniMax / MiMo).
PALETTE = {"hue": 349}
QUIRKS = [
    "attention-sink", "per-block-sizes", "sliding-window", "partial-rotary", "mixture-of-experts",
    "dense-first-blocks",
]

#: The checkpoints have 309B parameters: nothing here ran on real weights. Every identity, shape and snippet ran on
#: the pinned tiny checkpoint (block 0 full and dense, block 1 sliding and a mixture of 4 experts, top 2; queries and
#: keys 8 wide, values 16; float32); the sizes, the block layout and the rotary widths are the Hub configs', read on
#: meta builds.

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
            "detail": "{num_heads} heads, q·k {qk_head_dim}, v {v_head_dim}",
            "variants": {
                "full_attention": "{num_heads}×{qk_head_dim}/{v_head_dim}, {num_kv_heads} kv, full",
                "sliding_attention": "{num_heads}×{qk_head_dim}/{v_head_dim}, sink, window {sliding_window}",
            },
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
                "expert_outputs", "routed_output",
            ],
            "detail": "{num_experts} experts × {moe_intermediate_size}, top {top_k}",
        },
    ],
}

#: Notes on the model-level strip, by node: none, since nothing happens at the embeddings, the final norm or the head.
STRIP: dict[str, str] = {}

NOTES = """
## On sliding blocks `attention_scores` comes before the sink

Each sliding block's softmax takes one learned sink logit per head as an extra key column, and
`attention_scores` is read at the masked scores before that column joins, so `softmax(attention_scores)`
is not the pattern there. An edit to `attention_scores` is renormalized against the sink; an edit to
`attention_probabilities` sets the mix directly. Full blocks have no sink.

## `attention_values` carries `attention_value_scale`

The attention multiplies `v_proj`'s output by `attention_value_scale` (0.707) before the interface,
so `attention_values` is the scaled tensor. `self_attn.value_weight` includes the scale, and
`v_proj.weight` does not.

## The router chooses by the sigmoid plus a selection bias

The top `router_logits` are not always the chosen experts, so read the choice from `expert_indices`.
`expert_weights` are the chosen sigmoids without the bias, renormalized to sum to one, and there is
no shared expert.
"""
