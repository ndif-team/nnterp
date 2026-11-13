"""Qwen2-MoE: Qwen1.5-MoE and Qwen2-57B-A14B, Qwen2's attention with a mixture and a gated shared expert."""

MODEL_TYPE = "qwen2_moe"
TITLE = "Qwen1.5-MoE / Qwen2-MoE"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "Qwen/Qwen1.5-MoE-A2.7B"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-Qwen2MoeForCausalLM"
CHECKPOINTS = [
    "Qwen/Qwen1.5-MoE-A2.7B", "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    "Qwen/Qwen2-57B-A14B", "Qwen/Qwen2-57B-A14B-Instruct",
]

#: Set by hues.py (lineage: Qwen).
PALETTE = {"hue": 285}
QUIRKS = ["qkv-bias", "mixture-of-experts", "unnormalized-routing"]

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
            "detail": "{num_heads} heads, {num_kv_heads} kv; q, k, v biased",
        },
        {
            "host": "mlp",
            "kind": "moe",
            "part_order": ["shared", "router", "experts"],
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
## The shared expert is gated by a sigmoid

```
y   = post_attention_layernorm(h)
out = h + experts(y, router(y)) + sigmoid(shared_expert_gate(y)) * shared_experts(y)
```

Every token runs through the shared expert, and the mixture multiplies its output by a sigmoid gate,
one scalar per token. `shared_expert_output` is that product, the term the mixture adds beside
`routed_output`; `shared_experts.output` is the expert's output before the gate. Measure or patch the
shared expert's contribution at `shared_expert_output`.
"""
