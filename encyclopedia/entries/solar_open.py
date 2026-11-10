"""Solar Open: Llama's block with grouped-query attention under YaRN, and DeepSeek-V3's mixture with a shared expert on every block."""

MODEL_TYPE = "solar_open"
TITLE = "Solar Open"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "upstage/Solar-Open-100B"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "onnx-internal-testing/tiny-random-SolarOpenForCausalLM"
#: Upstage's one release with this config; the quantized copies on the Hub are other publishers'.
CHECKPOINTS = ["upstage/Solar-Open-100B"]

#: Set by hues.py (no kin, in a gap between lineages).
PALETTE = {"hue": 278}
QUIRKS = ["mixture-of-experts"]

#: Every number in the notes is Solar-Open-100B's config or a value transformers computes from it; no real
#: weights were run (about 100B parameters). The snippets ran on the pinned tiny checkpoint (4 experts, top 2),
#: with its e_score_correction_bias moved off zero for the router check.

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
            "detail": "{num_heads} heads over {num_kv_heads} kv, head_dim {head_dim}",
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
## A selection bias picks the experts but does not weigh them

```
s              = sigmoid(router_logits)
expert_indices = topk(s + e_score_correction_bias, 8)          # the bias picks
expert_weights = s[expert_indices] / sum(s[expert_indices])    # the bias does not weigh
mlp_output     = routed_output + shared_expert_output
```

`e_score_correction_bias` is added to the sigmoids only to choose, so the top `router_logits` are not
always the chosen experts. The chosen experts' weights are their unbiased sigmoids, renormalized to
sum to one.
"""
