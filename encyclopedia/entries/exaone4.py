"""EXAONE 4.0: a post-norm block with per-head query/key norms, and on 32B sliding blocks beside full blocks without rotary."""

MODEL_TYPE = "exaone4"
TITLE = "EXAONE 4.0"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "LGAI-EXAONE/EXAONE-4.0-1.2B"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-Exaone4ForCausalLM"
CHECKPOINTS = [
    "LGAI-EXAONE/EXAONE-4.0-1.2B",
    "LGAI-EXAONE/EXAONE-4.0-32B",
    "LGAI-EXAONE/EXAONE-4.0.1-32B",
]

#: Set by hues.py (no kin, in a gap between lineages).
PALETTE = {"hue": 182}
QUIRKS = ["post-norms", "qk-norm", "sliding-window", "nope-blocks"]

#: What the visualization draws: the sublayers in forward order, each with its norms,
#: the standard value that is its contribution, and the values read inside it.
#: ``detail`` and ``variants`` are formatted with the sizes and the config keys.
BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "post_norm": "post_attention_layernorm",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} query heads over {num_kv_heads} key/value heads, head_dim {head_dim}; "
                      "q_norm and k_norm per head, before the rotary",
            "variants": {
                "sliding_attention": "sliding window, RoPE",
                "full_attention": "full causal; NoPE on 32B",
            },
            "post_norm_note": "Follows the attention here; on Llama the same name is the MLP's pre-norm.",
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

#: Notes on the model-level strip, by node.
STRIP: dict[str, str] = {}

NOTES = """
## The contributions are the post-norms' outputs

```
h   = x + post_attention_layernorm(self_attn(x))
out = h + post_feedforward_layernorm(mlp(h))
```

`attention_output` and `mlp_output` are the outputs of the norms after the sublayers, the tensors the
block adds. The norm undoes a scale of the module's output, except in the first blocks of 1.2B, where
that output is small next to the norm's `eps`. Scale, steer and decompose through `attention_output`
and `mlp_output`.

## The tokenizer adds no beginning-of-sequence token

The sink that trained models form at position 0 falls on the prompt's first token. Leave position 0
out of activation statistics and steering-vector norms.
"""
