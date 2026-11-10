"""AFMoE (Arcee's Trinity): a sandwich block, gated attention with rotary on its sliding blocks only, dense first
blocks, then a sigmoid-routed mixture with a shared expert."""

MODEL_TYPE = "afmoe"
TITLE = "Trinity Nano / Trinity Mini / Trinity Large"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "arcee-ai/Trinity-Nano-Base"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-AfmoeForCausalLM"
CHECKPOINTS = [
    "arcee-ai/Trinity-Nano-Base", "arcee-ai/Trinity-Nano-Base-Pre-Anneal", "arcee-ai/Trinity-Nano-Preview",
    "arcee-ai/Trinity-Mini-Base", "arcee-ai/Trinity-Mini-Base-Pre-Anneal", "arcee-ai/Trinity-Mini",
    "arcee-ai/Trinity-Large-TrueBase", "arcee-ai/Trinity-Large-Base", "arcee-ai/Trinity-Large-Preview",
    "arcee-ai/Trinity-Large-Thinking",
]

#: Set by hues.py (no kin, in a gap between lineages).
PALETTE = {"hue": 230}
QUIRKS = [
    "sandwich-norms", "embedding-multiplier", "qk-norm", "output-gate", "sliding-window", "nope-blocks",
    "mixture-of-experts", "dense-first-blocks",
]

#: The smallest checkpoint, Trinity Nano, has about 6B parameters, so nothing here ran on real weights. Every
#: identity, shape, read order and snippet ran on the pinned tiny checkpoint (block 0 sliding and dense,
#: block 1 full and a mixture of 4 experts, top 2, two shared experts' width), with the config overridden at
#: load where a note says so (mup_enabled, sliding_window, a hand-written expert_bias). Sizes, scales and
#: block layouts are the Hub configs', read on meta builds.

BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "post_norm": "post_attention_layernorm",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} heads, {num_kv_heads} kv, q/k normed, gated",
            "variants": {
                "sliding_attention": "window {sliding_window}, rotary, gated",
                "full_attention": "full causal, no rotary, gated",
            },
            "post_norm_note": "Follows the attention here; on Llama the same name is the MLP's pre-norm.",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "pre_mlp_layernorm",
            "post_norm": "post_mlp_layernorm",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
        {
            "host": "mlp",
            "kind": "moe",
            "part_order": ["router", "shared", "experts"],
            "label": "MoE",
            "pre_norm": "pre_mlp_layernorm",
            "post_norm": "post_mlp_layernorm",
            "contribution": "mlp_output",
            "interior": [
                "router_logits", "expert_weights", "expert_indices",
                "expert_outputs", "routed_output", "shared_expert_output",
            ],
            "detail": "{num_experts} experts × {moe_intermediate_size}, top {top_k}",
            "post_norm_note": "Norms the shared and routed experts' sum, so mlp_output is not routed_output + shared_expert_output.",
        },
    ],
}

STRIP = {
    "embed": "The model multiplies it by √hidden_size before block 0 (mup_enabled), so layers[0].input is "
             "token_embeddings · √hidden_size.",
}

NOTES = """
## The contributions are the post-norms' outputs

```
h   = x + post_attention_layernorm(self_attn(input_layernorm(x)))
out = h + post_mlp_layernorm(mlp(pre_mlp_layernorm(h)))
```

`attention_output` and `mlp_output` are the outputs of the norms after the sublayers, the tensors the
block adds. RMSNorm is scale-invariant, so scaling `mlp.output` does nothing to the stream. Scale,
steer and decompose through `attention_output` and `mlp_output`. On a mixture block the norm takes
the shared and routed experts' sum, so an edit to one expert reaches the stream rescaled with the
token's whole MLP term.

## The expert weights sum to `route_scale`

The router picks experts by the sigmoid of `router_logits` plus a frozen selection bias,
`mlp.expert_bias`, so the top `router_logits` are not always the chosen experts. `expert_weights`
are the chosen sigmoids without the bias, renormalized and multiplied by `route_scale`, so they sum
to it at every token.
"""
