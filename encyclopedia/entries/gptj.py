"""GPT-J: EleutherAI's GPT-J-6B and the fine-tunes that load as GPTJForCausalLM."""

MODEL_TYPE = "gptj"
TITLE = "GPT-J"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "EleutherAI/gpt-j-6b"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-GPTJForCausalLM"
CHECKPOINTS = [
    "EleutherAI/gpt-j-6b",
    "togethercomputer/GPT-JT-6B-v1",
    "nomic-ai/gpt4all-j",
    "PygmalionAI/pygmalion-6b",
]

#: Set by hues.py (lineage: GPT-J).
PALETTE = {"hue": 54}
QUIRKS = ["parallel-blocks", "partial-rotary", "interleaved-rotary",
          "layernorm"]

#: What the visualization draws: one ln_1 whose output both sublayers read (drawn once per branch,
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
            "detail": "{num_heads} heads × {head_dim}, rotary {rotary_dim}",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "input_layernorm",
            "pre_norm_note": "Both sublayers read this one LayerNorm's output, the same tensor.",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {activation_function}",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "head": "lm_head adds a bias.",
}

NOTES = """
## The MLP does not read the attention

```
h   = input_layernorm(x)
out = self_attn(h) + mlp(h) + x
```

Both sublayers read the same normed block input, so ablating or patching a block's attention leaves
that block's MLP unchanged, and the first MLP to read block `i`'s attention is block `i + 1`'s.
`self_attn.input` and `mlp.input` are one tensor, so an in-place edit of `self_attn.input` reaches
the MLP too.
"""
