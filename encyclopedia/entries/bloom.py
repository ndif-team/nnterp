"""BLOOM: BigScience's BLOOM and BLOOMZ, every checkpoint that loads as BloomForCausalLM."""

MODEL_TYPE = "bloom"
TITLE = "BLOOM / BLOOMZ"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "bigscience/bloom-560m"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-BloomForCausalLM"
CHECKPOINTS = [
    "bigscience/bloom-560m", "bigscience/bloom-1b1", "bigscience/bloom-1b7",
    "bigscience/bloom-3b", "bigscience/bloom-7b1", "bigscience/bloom",
    "bigscience/bloomz-560m", "bigscience/bloomz-1b1", "bigscience/bloomz-1b7",
    "bigscience/bloomz-3b", "bigscience/bloomz-7b1", "bigscience/bloomz-7b1-mt",
    "bigscience/bloomz", "bigscience/bloomz-mt",
]

#: Set by hues.py (no kin, in a gap between lineages).
PALETTE = {"hue": 46}
QUIRKS = ["alibi", "qkv-bias", "layernorm"]

#: What the visualization draws: a sequential block with two LayerNorm pre-norms; each module adds
#: the residual itself, and its contribution is the tensor entering that add.
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
            "detail": "{num_heads} heads × {head_dim}, ALiBi",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "Reads self_attn.output[0], which is the stream after the attention's add.",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, tanh GELU",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "word_embeddings_layernorm normalizes it before block 0, so layers[0].input is not token_embeddings.",
}

NOTES = """
## Each module adds the residual itself

```
h   = self_attn(input_layernorm(x), residual=x)        # returns x + attention_output
out = mlp(post_attention_layernorm(h), residual=h)     # returns h + mlp_output
```

`self_attn.output[0]` and `mlp.output` are the stream after each add. Zeroing `self_attn.output[0]`
zeroes the stream, and every later module sees zeros; zeroing `attention_output` removes the
attention's contribution alone. Ablate, patch and steer through `attention_output` and `mlp_output`.

## A LayerNorm sits between the embedding and block 0

`layers[0].input` is `token_embeddings` through `word_embeddings_layernorm`, which has no standard
name. An edit at `token_embeddings` is renormalized with the rest of its row; edit
`layers[0].input` to set what block 0 reads.
"""
