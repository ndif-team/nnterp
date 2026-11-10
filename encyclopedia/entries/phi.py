"""Phi 1 / 1.5 / 2: Microsoft's small models that load as PhiForCausalLM."""

MODEL_TYPE = "phi"
TITLE = "Phi-1 / Phi-1.5 / Phi-2"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "microsoft/phi-2"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-PhiForCausalLM"
CHECKPOINTS = ["microsoft/phi-1", "microsoft/phi-1_5", "microsoft/phi-2"]

#: Set by hues.py (lineage: Phi).
PALETTE = {"hue": 137}
QUIRKS = ["parallel-blocks", "partial-rotary", "qkv-bias", "layernorm"]

#: Every real value in the notes was measured on microsoft/phi-2 (bfloat16 on CPU unless the note says float16,
#: which ran on a GPU) and, where named, microsoft/phi-1_5; the snippets also ran on the pinned tiny checkpoint.

#: What the visualization draws: one input_layernorm feeds both sublayers; the schema draws a pre-norm per
#: sublayer, so it appears twice, as one node.
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
            "detail": "{num_heads} heads × {head_dim}, partial rotary",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "input_layernorm",
            "pre_norm_note": "Both sublayers read this one LayerNorm's output, the same tensor.",
            "contribution": "mlp_output",
            "detail": "fc1 {hidden_size} → {intermediate_size}, {hidden_act}, fc2 → {hidden_size}",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "head": "lm_head has a bias, which project_on_vocab applies.",
}

NOTES = """
## The MLP does not read the attention

```
h   = input_layernorm(x)
out = self_attn(h) + mlp(h) + x
```

Both sublayers read the same normed input, so ablating or patching a block's attention leaves that
block's MLP unchanged. The first MLP to read block `i`'s attention is block `i + 1`'s.

## Load phi-2 in bfloat16 or float32 for the attention interior

phi-2 is stored in float16, and under `attn_implementation="eager"` in float16 its query-key products
overflow in the last blocks, so every logit is NaN. The default `sdpa` load in float16 is finite; pass
`dtype=torch.bfloat16` or `torch.float32` with the eager load.

## The prompt's first token carries one very large entry

The tokenizer adds no beginning-of-sequence token, and the stream at the prompt's first token has a
single dimension far larger than any entry elsewhere. Leave position 0 out of norms and means over
positions, or read it on its own.
"""
