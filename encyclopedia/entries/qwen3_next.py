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

#: No checkpoint of this family is under 3B parameters, so no real-weight run backs the notes: shapes, identities and
#: read orders are from the pinned tiny checkpoint, sizes from the configs.

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
## The shared expert is gated per token

```
h   = x + mixer(input_layernorm(x))       # linear_attn, or self_attn on every fourth block
m   = post_attention_layernorm(h)
out = h + routed_experts(m) + sigmoid(shared_expert_gate(m)) * shared_experts(m)
```

`shared_expert_output` is the gated product, the term the mixture adds, and `shared_experts.output`
is the expert before its gate. Ablate or patch the shared expert at `shared_expert_output`.

## The DeltaNet's queries and keys are copied per value head

The 16 key heads' queries and keys are repeated up to the 32 value heads before the kernel, so
`attention_queries` and `attention_keys` are 32 heads wide and heads `2j` and `2j + 1` are copies of
key head `j`. An edit to one copy reaches only its own value head.

## The Thinking template ends where a reasoning trace starts

With `add_generation_prompt=True` the Thinking template adds `<think>\\n` after
`<|im_start|>assistant\\n`, so a next-token metric there measures the start of a reasoning trace. The
Instruct and Coder templates end at `<|im_start|>assistant\\n`.
"""
