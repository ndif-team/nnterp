"""Cohere: Command R and Command R+ (CohereForCausalLM)."""

MODEL_TYPE = "cohere"
TITLE = "Command R / Command R+"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
#: Every CohereLabs repository is gated; this is an ungated copy of c4ai-command-r-08-2024 (its card names that base).
REFERENCE = "unsloth/c4ai-command-r-08-2024"
#: The Hub org the index files the family under: the copy is unsloth's; the model is CohereLabs'.
ORG = "CohereLabs"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "trl-internal-testing/tiny-CohereForCausalLM"
CHECKPOINTS = [
    "CohereLabs/c4ai-command-r-v01",
    "CohereLabs/c4ai-command-r-plus",
    "CohereLabs/c4ai-command-r-08-2024",
    "CohereLabs/c4ai-command-r-plus-08-2024",
    # Ungated copies of the 08-2024 releases.
    "unsloth/c4ai-command-r-08-2024",
    "unsloth/c4ai-command-r-plus-08-2024",
]

#: Set by hues.py (lineage: Cohere).
PALETTE = {"hue": 174}
QUIRKS = [
    "parallel-blocks", "layernorm", "interleaved-rotary",
    {"slug": "qk-norm", "when": lambda config: config.use_qk_norm},  # Command R+
    "scaled-logits",
]

#: What the visualization draws: one input_layernorm whose output both sublayers read (drawn once per branch,
#: one node), no post-norms; the two contributions join the stream in one add.
BLOCK = {
    "topology": "parallel",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "pre_norm_note": "Both sublayers read this one LayerNorm's output, the same tensor.",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} query, {num_kv_heads} key/value heads",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "input_layernorm",
            "pre_norm_note": "Both sublayers read this one LayerNorm's output, the same tensor.",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "logits": "logits = lm_head.output × logit_scale, and project_on_vocab multiplies too.",
}

NOTES = """
## The MLP does not read the attention

```
h   = input_layernorm(x)
out = x + self_attn(h) + mlp(h)
```

Both sublayers read the one norm's output, so ablating or patching a block's attention leaves that
block's MLP unchanged. The first MLP to read block `i`'s attention is block `i + 1`'s.

## The logits are the head's output times `logit_scale`

`logit_scale` is 0.0625 on Command R and 0.8333 on Command R+. `logits` is the scaled tensor and
`project_on_vocab` multiplies too, so a lens at the last block equals `logits`. A softmax or a
direct logit attribution taken on `lm_head.output` is off by that factor.
"""
