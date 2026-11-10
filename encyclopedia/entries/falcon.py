"""Falcon: TII's Falcon-7B, 11B, 40B, 180B and Falcon-RW, the three block layouts of FalconForCausalLM."""

MODEL_TYPE = "falcon"
TITLE = "Falcon"
SUBTITLE = (
    "A parallel block on every checkpoint but Falcon-RW: the block adds the attention's output into the MLP's "
    "output tensor in place, so mlp_output is a copy taken as the MLP returns."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "tiiuae/falcon-7b"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "Rocketknight1/tiny-random-falcon-7b"
#: Three layouts: 7B and 11B (parallel, one input_layernorm), 40B and 180B (parallel, ln_attn and ln_mlp), Falcon-RW
#: (sequential, input_layernorm and post_attention_layernorm, ALiBi). BLOCK picks one from each config.
CHECKPOINTS = [
    "tiiuae/falcon-7b", "tiiuae/falcon-7b-instruct",
    "tiiuae/falcon-11B",
    "tiiuae/falcon-40b", "tiiuae/falcon-40b-instruct", "tiiuae/falcon-180B",
    "tiiuae/falcon-rw-1b", "tiiuae/falcon-rw-7b",
]

#: Set by hues.py (lineage: Falcon).
PALETTE = {"hue": 17}
QUIRKS = [
    {"slug": "parallel-blocks", "when": lambda config: config.parallel_attn},  # all but Falcon-RW
    {"slug": "qkv-bias", "when": lambda config: config.bias},  # Falcon-RW
    "layernorm",
    {"slug": "alibi", "when": lambda config: config.alibi},  # Falcon-RW
]

INTERIOR = [
    "attention_queries", "attention_keys", "attention_values",
    "attention_scores", "attention_probabilities", "attention_head_outputs",
]
#: 7B and 11B: one input_layernorm whose output both sublayers read (drawn once per branch, one node), no
#: post-norms; the two contributions join the stream in one add.
SEVEN_B = {
    "topology": "parallel",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "pre_norm_note": "Both sublayers read this one LayerNorm's output, the same tensor.",
            "contribution": "attention_output",
            "interior": INTERIOR,
            "detail": "{num_heads} heads × {head_dim}, {num_kv_heads} kv",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "input_layernorm",
            "pre_norm_note": "Both sublayers read this one LayerNorm's output, the same tensor.",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, GELU",
        },
    ],
}

#: 40B and 180B (new_decoder_architecture, two norms in parallel): each sublayer reads its own LayerNorm of the
#: block input, ln_attn and ln_mlp, and the two contributions join the stream in one add.
FORTY_B = {
    "topology": "parallel",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "ln_attn",
            "contribution": "attention_output",
            "interior": INTERIOR,
            "detail": "{num_heads} heads × {head_dim}, {num_kv_heads} kv",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "ln_mlp",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, GELU",
        },
    ],
}

#: Falcon-RW: sequential, input_layernorm before the attention and post_attention_layernorm before the MLP.
RW = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "contribution": "attention_output",
            "interior": INTERIOR,
            "detail": "{num_heads} heads × {head_dim}, ALiBi",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, GELU",
        },
    ],
}

#: What the visualization draws, by the checkpoint's config: Falcon-RW is the one sequential layout; a
#: new_decoder_architecture config with two norms in parallel is 40B's (11B sets num_ln_in_parallel_attn to 1).
BLOCK = [
    (lambda config: not config.parallel_attn, RW),
    (lambda config: config.new_decoder_architecture and config.num_ln_in_parallel_attn != 1, FORTY_B),
    (lambda config: True, SEVEN_B),
]

#: Notes on the model-level strip, by node: none, since nothing happens at the embeddings, the final norm or the head.
STRIP: dict[str, str] = {}

NOTES = """
## The MLP does not read the attention

```
a   = self_attn(input_layernorm(x))   # ln_attn(x) on 40B and 180B
m   = mlp(input_layernorm(x))         # ln_mlp(x) on 40B and 180B
out = (m + a) + x                     # a is added into m in place
```

On every checkpoint but Falcon-RW both sublayers read the block's input, so ablating or patching a
block's attention leaves that block's MLP unchanged. The first MLP to read block `i`'s attention is
block `i + 1`'s.

## `mlp.output` holds the attention's output too

On the same checkpoints the block adds the attention's output into the MLP's output tensor in place,
so a saved `mlp.output` reads the sum of the two. `mlp_output` is the MLP's contribution alone.

## On Falcon-RW the eager attention adds ALiBi twice

transformers builds the ALiBi bias into the attention mask, and the eager path adds it to the scores
again, so `attention_scores` carry it twice where the default sdpa load adds it once. An eager load,
which the attention interior needs, runs a different model: on `falcon-rw-1b` in float32 the two
loads' logits differ by up to about 1.5.

## On 11B, 40B and 180B the keys and values are copied per query head

The attention copies each key/value head to its query heads before nnterp reads them, so
`attention_keys` and `attention_values` are `num_heads` wide although `num_kv_heads` is smaller. An
edit to one copy reaches only that query head.
"""
