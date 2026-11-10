"""GPT-OSS: Llama's pre-norm block with a learned attention sink on every head, sliding and full blocks
alternating, and a mixture of experts whose experts are clamped SwiGLUs with biases."""

MODEL_TYPE = "gpt_oss"
TITLE = "GPT-OSS"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "openai/gpt-oss-20b"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "yujiepan/gpt-oss-tiny-random"
CHECKPOINTS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]

#: Set by hues.py (no kin, in a gap between lineages).
PALETTE = {"hue": 148}
QUIRKS = ["attention-sink", "sliding-window", "mixture-of-experts", "qkv-bias"]

#: The real checkpoints are 21B and 117B parameters: nothing here ran on real weights. Every identity and
#: shape in the notes, and every snippet, ran on the pinned tiny checkpoint (2 blocks, 32 experts, top 4);
#: the sizes are the 20b and 120b configs'.

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
            "detail": "{num_heads} heads × {head_dim}, {num_kv_heads} kv, sink",
            "variants": {
                "sliding_attention": "{num_heads}×{head_dim}, sink, window {sliding_window}",
                "full_attention": "{num_heads}×{head_dim}, sink, full causal",
            },
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

STRIP: dict[str, str] = {}

NOTES = """
## `softmax(attention_scores)` is not the pattern

`attention_scores` is read before each head's learned sink joins the softmax, so its softmax
overstates every key, and `1 - attention_probabilities.sum(-1)` is the sink's share of each row. An
edit to `attention_scores` is renormalized against the sink; an edit to `attention_probabilities` is
not.

## `expert_outputs` are weighted and sum to `mlp_output`

The router softmaxes its top four logits alone, so a token's `expert_weights` sum to one. Each
`expert_outputs` slot is an expert's output, its `down_proj` bias included, times its weight, and
with no shared expert the slots sum to `mlp_output`.
"""
