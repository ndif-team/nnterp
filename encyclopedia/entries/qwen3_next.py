"""Qwen3-Next: a gated DeltaNet hybrid with a mixture of experts on every block, text only."""

MODEL_TYPE = "qwen3_next"
TITLE = "Qwen3-Next / Qwen3-Coder-Next"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "Qwen/Qwen3-Next-80B-A3B-Instruct"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "yujiepan/qwen3-next-moe-tiny-random"
#: The FP8 and GGUF copies are left out.
CHECKPOINTS = [
    "Qwen/Qwen3-Next-80B-A3B-Instruct", "Qwen/Qwen3-Next-80B-A3B-Thinking",
    "Qwen/Qwen3-Coder-Next", "Qwen/Qwen3-Coder-Next-Base",
]

#: Set by hues.py (lineage: Qwen).
PALETTE = {"hue": 307}
QUIRKS = ["hybrid", "gated-query", "mixture-of-experts", "qk-norm", "partial-rotary", "gain-norm"]

#: The sublayers in forward order. A block draws the mixer it has (``linear_attn`` or ``self_attn``); every block of
#: every checkpoint, and of the pinned tiny, has the mixture.
BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "linear_attn",
            "kind": "mixer",
            "label": "Linear attention",
            "pre_norm": "input_layernorm",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values", "decays", "betas",
                "state_input", "attention_head_outputs", "state_output", "states",
            ],
            "detail": "gated DeltaNet, {linear_num_value_heads} heads × {linear_value_head_dim}",
        },
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
            "detail": "{num_heads} heads, {num_kv_heads} kv, output-gated",
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
            "detail": "{num_experts} experts × {moe_intermediate_size}, top {top_k}, gated shared",
        },
    ],
}

#: Notes on the model-level strip, by node: none, since nothing happens at the embeddings, the final norm or the head.
STRIP: dict[str, str] = {}

NOTES = """
## The shared expert is gated by a sigmoid

```
h   = x + mixer(input_layernorm(x))       # linear_attn, or self_attn on every fourth block
y   = post_attention_layernorm(h)
out = h + experts(y, router(y)) + sigmoid(shared_expert_gate(y)) * shared_experts(y)
```

Every token runs through the shared expert, and the mixture multiplies its output by a sigmoid gate, one
scalar per token. `shared_expert_output` is that product, the term the mixture adds beside
`routed_output`; `shared_experts.output` is the expert's output before the gate. Measure or patch the
shared expert's contribution at `shared_expert_output`.

## The DeltaNet's queries and keys are copied per value head

The mixer repeats each key head's query and key up to the value heads before the kernel, so
`attention_queries` and `attention_keys` have one head per value head, each key head copied into
consecutive heads. An edit to one copy reaches only its own value head; edit every copy to change the
key head.

## The Thinking template ends where a reasoning trace starts

With `add_generation_prompt=True` the Thinking template adds `<think>\\n` after
`<|im_start|>assistant\\n`, so a next-token metric there measures the start of a reasoning trace. The
Instruct and Coder templates end at `<|im_start|>assistant\\n`.
"""
