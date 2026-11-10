"""Qwen3-MoE: Qwen3's attention with a mixture of experts on every block."""

MODEL_TYPE = "qwen3_moe"
TITLE = "Qwen3-MoE"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "Qwen/Qwen3-30B-A3B"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "trl-internal-testing/tiny-Qwen3MoeForCausalLM"
CHECKPOINTS = [
    "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-30B-A3B-Base",
    "Qwen/Qwen3-30B-A3B-Instruct-2507", "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "Qwen/Qwen3-235B-A22B", "Qwen/Qwen3-235B-A22B-Instruct-2507", "Qwen/Qwen3-235B-A22B-Thinking-2507",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct", "Qwen/Qwen3-Coder-480B-A35B-Instruct",
]

#: Set by hues.py (lineage: Qwen).
PALETTE = {"hue": 298}
QUIRKS = ["qk-norm", "mixture-of-experts"]

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
            "detail": "{num_heads} heads, {num_kv_heads} kv; q, k RMS-normed",
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
## The chat template ends where a reasoning trace starts

On 30B-A3B, 30B-A3B-Base and 235B-A22B `add_generation_prompt=True` ends the prompt at
`<|im_start|>assistant\\n` and the next token is `<think>`, so a next-token metric there
measures the start of a reasoning trace. Pass `enable_thinking=False`, which appends an empty
thinking block, to put the answer next. The Thinking-2507 templates ignore the flag and always open
the block; the Instruct-2507 and Coder templates have none.

## Scaling a head's slice of `q_proj.output` does nothing

`q_norm` and `k_norm` are RMSNorms over each head's dimensions, applied after the projections, so
they undo a scale on a head's slice of `q_proj.output` or `k_proj.output`. To sharpen or flatten a
head's pattern, edit `attention_queries` or `attention_keys`, which are read after the norms.
"""
