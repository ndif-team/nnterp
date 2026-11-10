"""Falcon-H1: a parallel hybrid, a Mamba-2 mixer and attention side by side in every block, then an MLP."""

MODEL_TYPE = "falcon_h1"
TITLE = "Falcon-H1"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "tiiuae/Falcon-H1-0.5B-Base"
#: The tiny checkpoint the test suite builds the page from (the released multipliers; loaded in float32 by the suite).
PINNED = "yujiepan/falcon-h1-tiny-random"
CHECKPOINTS = [
    "tiiuae/Falcon-H1-0.5B-Base", "tiiuae/Falcon-H1-0.5B-Instruct",
    "tiiuae/Falcon-H1-1.5B-Base", "tiiuae/Falcon-H1-1.5B-Instruct",
    "tiiuae/Falcon-H1-1.5B-Deep-Base", "tiiuae/Falcon-H1-1.5B-Deep-Instruct",
    "tiiuae/Falcon-H1-3B-Base", "tiiuae/Falcon-H1-3B-Instruct",
    "tiiuae/Falcon-H1-7B-Base", "tiiuae/Falcon-H1-7B-Instruct",
    "tiiuae/Falcon-H1-34B-Base", "tiiuae/Falcon-H1-34B-Instruct",
    "tiiuae/Falcon-H1R-7B",
    "tiiuae/Falcon-H1-Tiny-90M-Base", "tiiuae/Falcon-H1-Tiny-90M-Instruct",
]

#: Set by hues.py (lineage: Falcon).
PALETTE = {"hue": 14}
QUIRKS = ["parallel-mixers", "mamba2", "scaled-residual-adds", "embedding-multiplier", "scaled-logits"]

NORM_NOTE = "One norm, drawn on both mixers' rows: both read its output."

#: The block in forward order: the Mamba-2 mixer and the attention both read input_layernorm's output and join the
#: stream at one add, then the MLP reads the stream after that add. `parallel_with_next` marks the mixer as the
#: first branch of that pair, so the page draws the two mixers side by side, joining at one add.
BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "linear_attn",
            "kind": "mixer",
            "label": "Mamba-2",
            "pre_norm": "input_layernorm",
            "pre_norm_note": NORM_NOTE,
            "parallel_with_next": True,
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values", "betas", "decays",
                "state_input", "attention_head_outputs", "state_output", "states",
            ],
            "detail": "SSD, beside self_attn",
        },
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "pre_norm_note": NORM_NOTE,
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads}/{num_kv_heads} heads × {head_dim}",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "Native pre_ff_layernorm, the MLP's input norm.",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, gated {hidden_act}",
        },
    ],
    "identity": "layers[i].input + (linear_attn.attention_output + self_attn.attention_output) + mlp.mlp_output == layer_output",
}

STRIP = {
    "embed": "The model multiplies embed_tokens' output by embedding_multiplier before block 0: token_embeddings is the "
             "unscaled lookup, layers[0].input the scaled one.",
    "logits": "lm_head.output times lm_head_multiplier; project_on_vocab applies it.",
}

NOTES = """
## Each mixer's contribution is its output times a multiplier

```
n   = input_layernorm(x)
h   = x + mamba(n) * ssm_out_multiplier + self_attn(n) * attention_out_multiplier
out = h + mlp(post_attention_layernorm(h))
```

Each mixer's `attention_output` is the module's output times its multiplier, the tensor the block
adds; `linear_attn.output` and `self_attn.output[0]` are unscaled. Patch, ablate and decompose
through the two `attention_output` values.

## The prompt's first token is an attention sink

The tokenizer adds no beginning-of-sequence token, so the prompt's first token takes much of the
attention, and in the middle blocks its stream has a norm many times the other positions'. Leave
position 0 out of activation statistics and steering-vector norms.
"""
