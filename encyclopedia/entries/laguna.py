"""Laguna: each block with its own number of query heads, an attention gated per head after the interface, and a
mixture with a shared expert after a dense first block."""

MODEL_TYPE = "laguna"
TITLE = "Laguna XS / Laguna S / Laguna M"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "poolside/Laguna-XS-2.1"
#: The tiny checkpoint the test suite builds the page from (two heads on both blocks).
PINNED = "hf-tiny-v2/tiny-random-LagunaForCausalLM"
#: The FP8, INT4 and NVFP4 copies and the DFlash drafters (DFlashLagunaForCausalLM) are left out.
CHECKPOINTS = [
    "poolside/Laguna-XS-2.1", "poolside/Laguna-XS.2", "poolside/Laguna-S-2.1",
    "poolside/Laguna-M.1", "poolside/Laguna-M.1-base",
]

#: Set by hues.py (no kin, in a gap between lineages).
PALETTE = {"hue": 194}
QUIRKS = [
    # XS and S: 48 query heads on the full blocks, 64 or 72 on the sliding ones; M.1: 64 on every block
    {"slug": "per-block-sizes", "when": lambda config: len(set(config.num_attention_heads_per_layer)) > 1},
    "qk-norm", "output-gate",
    # every block of M.1 is full_attention
    {"slug": "sliding-window", "when": lambda config: "sliding_attention" in config.layer_types},
    # partial_rotary_factor 0.5 on XS and S's full blocks; 1.0 (the whole head) on M.1
    {"slug": "partial-rotary", "when": lambda config: any(
        p.get("partial_rotary_factor", 1.0) < 1.0 for p in config.rope_parameters.values())},
    "mixture-of-experts", "dense-first-blocks",
]

#: The sublayers in forward order; a block draws the MLP its ``mlp`` is (dense on the first blocks, a mixture after).
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
            "detail": "{num_kv_heads} kv × {head_dim}, q/k normed, gated",
            "variants": {
                "full_attention": "{num_heads} heads × {head_dim}, full, gated",
                "sliding_attention": "{num_kv_heads} kv, window {sliding_window}, gated",
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
            "part_order": ["shared", "router", "experts"],
            "label": "MoE",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "interior": [
                "shared_expert_output", "router_logits", "expert_weights", "expert_indices",
                "expert_outputs", "routed_output",
            ],
            "detail": "{num_experts} experts × {moe_intermediate_size}, top {top_k}, 1 shared",
        },
    ],
}

#: Notes on the model-level strip, by node: none, since nothing happens at the embeddings, the final norm or the head.
STRIP: dict[str, str] = {}

NOTES = """
## `expert_outputs` are read before `moe_routed_scaling_factor`

`routed_output` is the experts' weighted sum times `moe_routed_scaling_factor` (2.5 on XS and S, 1.0
on M.1), so a slot adds that factor times its `expert_outputs` entry to the stream. The router
chooses by the sigmoid plus a selection bias, so the top `router_logits` are not always the chosen
experts; `expert_weights` are the chosen sigmoids alone, renormalized to sum to one. A shared expert
runs on every token, and `mlp_output` is `routed_output + shared_expert_output`.
"""
