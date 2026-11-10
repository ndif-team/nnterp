"""DeepSeek-V3: V2's latent attention, three dense first blocks, and a sigmoid-routed mixture with a selection bias."""

MODEL_TYPE = "deepseek_v3"
TITLE = "DeepSeek-V3 / DeepSeek-R1 / Moonlight"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "deepseek-ai/DeepSeek-V3"
#: The tiny checkpoint; the suite rewrites its config (first_k_dense_replace 1, topk_group 1) into a local copy.
PINNED = "hf-internal-testing/tiny-random-DeepseekV3ForCausalLM"
CHECKPOINTS = [
    "deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-V3-Base", "deepseek-ai/DeepSeek-V3-0324",
    "deepseek-ai/DeepSeek-R1", "deepseek-ai/DeepSeek-R1-Zero", "deepseek-ai/DeepSeek-R1-0528",
    "deepseek-ai/DeepSeek-V3.1", "deepseek-ai/DeepSeek-V3.1-Base", "deepseek-ai/DeepSeek-V3.1-Terminus",
    "moonshotai/Moonlight-16B-A3B", "moonshotai/Moonlight-16B-A3B-Instruct",
]

#: Set by hues.py (lineage: DeepSeek).
PALETTE = {"hue": 325}
QUIRKS = ["latent-attention", "interleaved-rotary", "mixture-of-experts", "dense-first-blocks"]

#: Shapes, identities, read orders and the routing were run on the suite's copy of the pinned tiny checkpoint
#: (block 0 dense, block 1 a mixture of 4 experts, top 2; queries and keys 192 wide, values 128), with a
#: selection bias written by hand for the bias claims. Sizes, scales and routing settings are the Hub
#: configs', read on meta builds; no real weights were run (the smallest checkpoint, Moonlight, has 16B).

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

#: Notes on the model-level strip, by node: none, since nothing happens at the embeddings, the final norm or the head.
STRIP: dict[str, str] = {}

NOTES = """
## Every head shares the rotary part of the keys

Keys are expanded per head from one latent per token, but their last `qk_rope_head_dim` dimensions
come from one projection shared by every head: `attention_keys[:, h, :, -qk_rope_head_dim:]` is the
same tensor for every `h`. An edit to that part acts as an edit to the shared key only when it is
made on every head. `config.head_dim` is the rotary part's width, which no served value has.

## The score scale carries YaRN's factor

The DeepSeek checkpoints extend their context with YaRN, and the attention multiplies the scores by
`qk_head_dim ** -0.5` times a factor set by `mscale_all_dim`. Recompute `attention_scores` from the
queries and keys with `self_attn.scaling`. Moonlight has no YaRN and no factor.

## The expert weights sum to `routed_scaling_factor`

The router picks experts by the sigmoid of `router_logits` plus a selection bias,
`router.e_score_correction_bias`, and on the DeepSeek checkpoints only from a token's best
`topk_group` groups of experts, so the top `router_logits` are not always the chosen experts.
`expert_weights` are the chosen sigmoids without the bias, renormalized and multiplied by
`routed_scaling_factor`, so they sum to it at every token.
"""
