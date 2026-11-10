"""Qwen3.5 and Qwen3.6, mixture of experts: the text model of the multimodal MoE checkpoints, a gated DeltaNet hybrid."""

MODEL_TYPE = "qwen3_5_moe_text"
TITLE = "Qwen3.5-MoE / Qwen3.6-MoE"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "Qwen/Qwen3.5-35B-A3B-Base"
#: The tiny checkpoint the test suite builds the page from: the wrapper itself, as every checkpoint is.
PINNED = "yujiepan/qwen3.5-moe-tiny-random"
#: Every checkpoint is the vision-language wrapper (model_type qwen3_5_moe); the FP8 and GPTQ copies are left out.
CHECKPOINTS = [
    "Qwen/Qwen3.5-35B-A3B-Base", "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.5-122B-A10B",
    "Qwen/Qwen3.5-397B-A17B",
    "Qwen/Qwen3.6-35B-A3B",
]

#: The vision-language wrapper every checkpoint above is, keyed by its config.model_type. The vision encoder comes from
#: the checkpoint's vision_config.model_type (encyclopedia/vision/qwen_vit.py); what a config does not say is here.
#: Shapes and identities were checked on the pinned tiny wrapper; the sizes are the checkpoints' configs.
WRAPPERS = {
    "qwen3_5_moe": {
        "title": "Qwen3.5-MoE",
        "pinned": "yujiepan/qwen3.5-moe-tiny-random",
        "projector": "merger, inside the vision encoder: a LayerNorm on each patch, then an MLP (linear_fc1, GELU, "
                     "linear_fc2) over each 2 × 2 block of patches concatenated, one image token per block",
        "projector_input": "the merger's input: the last block's output",
    },
}

#: Set by hues.py (lineage: Qwen).
PALETTE = {"hue": 313}
QUIRKS = ["hybrid", "gated-query", "mixture-of-experts", "qk-norm", "partial-rotary", "gain-norm", "multimodal-rotary"]

#: No checkpoint of this family is under 3B parameters, so no real-weight run backs the notes: shapes, identities and
#: read orders are from the pinned tiny checkpoint (as the text model and as the wrapper), sizes from the configs.

#: The sublayers in forward order. A block draws the mixer it has (``linear_attn`` or ``self_attn``);
#: every block has the mixture.
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

#: Notes on the model-level strip, by node.
STRIP: dict[str, str] = {}

NOTES = """
## The shared expert is gated by a sigmoid

```
h   = x + mixer(input_layernorm(x))               # linear_attn or self_attn
y   = post_attention_layernorm(h)
out = h + experts(y, router(y)) + sigmoid(shared_expert_gate(y)) * shared_experts(y)
```

Every token runs through the shared expert, and the mixture multiplies its output by a sigmoid gate,
one scalar per token. `shared_expert_output` is that product, the term the mixture adds beside
`routed_output`; `shared_experts.output` is the expert's output before the gate. Measure or patch the
shared expert's contribution at `shared_expert_output`.

## The DeltaNet's queries and keys are served once per value head

The mixer has fewer key heads than value heads, and it repeats the queries and keys up to the value
heads before the kernel, so `attention_queries` and `attention_keys` are `num_v_heads` wide, adjacent
heads being copies of one key head (`repeat_interleave`). An edit to one copy reaches only its value
head; edit every copy of a key head to change it.
"""
