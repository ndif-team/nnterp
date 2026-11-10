"""Phi-3.5-MoE and the SlimMoE checkpoints that load as PhimoeForCausalLM."""

MODEL_TYPE = "phimoe"
TITLE = "Phi-3.5-MoE"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "microsoft/Phi-3.5-MoE-instruct"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-PhimoeForCausalLM"
CHECKPOINTS = [
    "microsoft/Phi-3.5-MoE-instruct",
    "microsoft/Phi-mini-MoE-instruct", "microsoft/Phi-tiny-MoE-instruct",
]

#: Set by hues.py (lineage: Phi).
PALETTE = {"hue": 144}
QUIRKS = [
    "mixture-of-experts",
    # Every public checkpoint sets attention_bias; the pinned tiny one does not.
    {"slug": "qkv-bias", "when": lambda config: config.attention_bias},
    "layernorm",
    # Phi-3.5-MoE's window equals its context and never masks; the SlimMoE checkpoints' is 2047.
    {"slug": "sliding-window",
     "when": lambda config: config.sliding_window is not None and config.sliding_window < config.max_position_embeddings},
]

#: Every checkpoint is above 3B parameters, so nothing here ran on real weights: shapes, the routing arithmetic
#: and the snippets are from the pinned tiny checkpoint (16 experts, top 2) and a copy of it with
#: attention_bias and lm_head_bias set; the rotary from transformers' rotary module built on the reference
#: config; every size from the configs.

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
            "detail": "{num_heads} heads, {num_kv_heads} kv, biased",
        },
        {
            "host": "mlp",
            "kind": "moe",
            "label": "MoE",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "The mixture's input norm, named for its place after the attention's add.",
            "contribution": "mlp_output",
            "interior": [
                "router_logits", "expert_weights", "expert_indices",
                "expert_outputs", "routed_output",
            ],
            "detail": "{num_experts} experts × {intermediate_size}, top {top_k}",
        },
    ],
}

STRIP = {
    "head": "lm_head has a bias (lm_head_bias), which project_on_vocab applies.",
}

NOTES = """
## A token's two expert weights sum to up to 2

```
y   = post_attention_layernorm(h)
out = h + w0 * experts[i0](y) + w1 * experts[i1](y)
```

The router (`sparsemixer`) picks the top two logits one after the other. Each slot's weight is that
expert's share of a softmax over the logits within a small jitter threshold of the slot's maximum:
exactly 1 unless a near-tie shares it, and the share the near-tie takes is dropped. The two weights
are not renormalized together, so each chosen expert's output joins the stream at or near full weight.

## `" Paris"` is two tokens

The tokenizer is Phi-3's and splits a word after a space as `' Paris'` → `['▁', '▁Paris']`. Encode
`"Paris"` to get the word's one token as the answer token for a logit metric.
"""
