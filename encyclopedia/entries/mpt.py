"""MPT: MosaicML's MPT-7B and MPT-30B and the checkpoints that load as transformers' own MptForCausalLM."""

MODEL_TYPE = "mpt"
TITLE = "MPT"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "ybelkada/mpt-7b-bf16-sharded"
#: The Hub org the index files the family under: the copy is ybelkada's; the model is MosaicML's.
ORG = "mosaicml"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-MptForCausalLM"
#: mosaicml's own repositories are gone from the Hub; the copies below carry MPT-7B's and MPT-30B's configs.
CHECKPOINTS = [
    "ybelkada/mpt-7b-bf16-sharded",
    "anas-awadalla/mpt-7b",
    "gl198976/mpt-7b-instruct",
    "Alchan/mpt-7b-chat",
    "eluzhnica/mpt-30b-peft-compatible",
    "team-lucid/mptk-1b",
    "mosaicml/mpt-7b",
    "ethzanalytics/mpt-7b-storywriter-sharded",
]

#: Set by hues.py (no kin, in a gap between lineages).
PALETTE = {"hue": 86}
QUIRKS = ["alibi", "layernorm"]

#: What the visualization draws: a sequential block with two bias-free LayerNorm pre-norms; the block adds the
#: attention, the MLP module adds the residual itself.
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
            "detail": "{num_heads} heads × {head_dim}, Wqkv, ALiBi",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "The MLP's input norm, named for its place after the attention's add.",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, GELU",
        },
    ],
    "identity_note": "The MLP module makes the second add itself, so mlp.output is layer_output.",
}

#: Notes on the model-level strip, by node.
STRIP: dict[str, str] = {}

NOTES = """
## `mlp.output` is the block's output

```
h   = x + self_attn(input_layernorm(x))
out = mlp(post_attention_layernorm(h), residual=h)   # returns h + mlp_output
```

The MLP takes the stream as a second argument and adds it to its own output, so `mlp.output` equals
`layer_output`. Zeroing `mlp.output` zeroes the stream that leaves the block; ablate, steer and patch
the MLP through `mlp_output`.
"""
