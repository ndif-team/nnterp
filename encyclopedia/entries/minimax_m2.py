"""MiniMax-M2: Llama's pre-norm block with query and key norms over the whole projection, and a mixture of experts
on every block chosen by sigmoid scores plus a selection bias."""

MODEL_TYPE = "minimax_m2"
TITLE = "MiniMax-M2 / MiniMax-M2.1 / MiniMax-M2.5 / MiniMax-M2.7"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "MiniMaxAI/MiniMax-M2"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-MiniMaxM2ForCausalLM"
CHECKPOINTS = ["MiniMaxAI/MiniMax-M2", "MiniMaxAI/MiniMax-M2.1", "MiniMaxAI/MiniMax-M2.5", "MiniMaxAI/MiniMax-M2.7"]

#: Set by hues.py (lineage: MiniMax / MiMo).
PALETTE = {"hue": 346}
QUIRKS = [
    "qk-norm", "mixture-of-experts",
    # the Hub configs' rotary_dim 64 of head_dim 128, which MiniMaxM2Config reads as partial_rotary_factor 0.5
    {"slug": "partial-rotary", "when": lambda config: (config.rope_parameters or {}).get("partial_rotary_factor", 1.0) < 1.0},
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
            "detail": "{num_heads} heads × {head_dim}, {num_kv_heads} kv, q/k normed",
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
            "detail": "{num_experts} experts × {intermediate_size}, top {top_k}",
        },
    ],
}

#: Notes on the model-level strip, by node: none, since nothing happens at the embeddings, the final norm or the head.
STRIP: dict[str, str] = {}

NOTES = """
## `q_norm` and `k_norm` each span every head

`q_norm` is one RMSNorm over `q_proj`'s whole output and `k_norm` one over `k_proj`'s, applied before
the heads are split, so an edit to one head's slice of `q_proj.output` changes every head's queries.
Edit one head at `attention_queries` or `attention_keys`, which no norm follows.

## The routing slots are unsorted

The router asks `topk` for unsorted slots, so slot 0 need not hold the largest weight. Compare sets of
experts, not slots. It chooses by the sigmoid plus a selection bias, so the top `router_logits` are not
always the chosen experts. `expert_weights` are the chosen sigmoids without the bias, renormalized to
sum to one, and there is no shared expert.
"""
