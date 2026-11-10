"""OLMo 3: OLMo-2's post-norm block, with three sliding-window blocks for every full-attention block."""

MODEL_TYPE = "olmo3"
TITLE = "OLMo 3 / OLMo 3.1"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "allenai/Olmo-3-1025-7B"
#: The tiny checkpoint the test suite builds the page from (the test rewrites its config into a local copy).
PINNED = "yujiepan/olmo-3-tiny-random"
CHECKPOINTS = [
    "allenai/Olmo-3-1025-7B",
    "allenai/Olmo-3-7B-Instruct-SFT", "allenai/Olmo-3-7B-Instruct",
    "allenai/Olmo-3-7B-Think-SFT", "allenai/Olmo-3-7B-Think",
    "allenai/Olmo-3-7B-RL-Zero-Math",
    "allenai/Olmo-3-1125-32B", "allenai/Olmo-3-32B-Think",
    "allenai/Olmo-3.1-32B-Instruct", "allenai/Olmo-3.1-32B-Think",
]

#: Set by hues.py (lineage: OLMo).
PALETTE = {"hue": 101}
QUIRKS = ["post-norms", "qk-norm", "sliding-window"]

BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "post_norm": "post_attention_layernorm",
            "post_norm_note": "Follows the attention here; on Llama the same name is the MLP's pre-norm.",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} heads, q_norm and k_norm",
            "variants": {
                "sliding_attention": "window {sliding_window}, plain rotary",
                "full_attention": "full causal, YaRN rotary",
            },
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "post_norm": "post_feedforward_layernorm",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
    ],
}

STRIP: dict[str, str] = {}

NOTES = """
## The contributions are the post-norms' outputs

```
h   = x + post_attention_layernorm(self_attn(x))
out = h + post_feedforward_layernorm(mlp(h))
```

`attention_output` and `mlp_output` are the outputs of the norms after the sublayers, the tensors the
block adds. RMSNorm is scale-invariant, so halving `mlp.output` does nothing to the stream. Scale,
steer and decompose through `attention_output` and `mlp_output`.

## The full-attention blocks scale their queries and keys

The full blocks use YaRN rotary, which multiplies the cosines and sines by its `attention_factor`,
1.2079; the sliding blocks use the plain rotary. On a full block each query and key head has 1.2079
times the norm it had leaving `q_norm` or `k_norm`, and the scores 1.459 times, so compare query,
key and score magnitudes within one kind of block.

## `q_norm` and `k_norm` span every head

Each takes one RMS over the whole projection before the heads are split, so an edit to one head's
slice of `q_proj.output` or `k_proj.output` reaches the other heads. Edit one head at
`attention_queries` or `attention_keys`, which no norm follows.
"""
