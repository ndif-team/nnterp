"""OLMo 1: Llama's pre-norm block with LayerNorms that have no weight and no bias, and a clamp on queries, keys and values on some checkpoints."""

MODEL_TYPE = "olmo"
TITLE = "OLMo / OLMo 0424 / OLMo 0724"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "allenai/OLMo-1B-hf"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "katuni4ka/tiny-random-olmo-hf"
CHECKPOINTS = [
    "allenai/OLMo-1B-hf", "allenai/OLMo-7B-hf", "allenai/OLMo-7B-Twin-2T-hf",
    "allenai/OLMo-7B-SFT-hf", "allenai/OLMo-7B-Instruct-hf",
    "allenai/OLMo-7B-0424-hf", "allenai/OLMo-7B-0424-SFT-hf",
    "allenai/OLMo-1B-0724-hf", "allenai/OLMo-7B-0724-hf",
    "allenai/OLMo-7B-0724-SFT-hf", "allenai/OLMo-7B-0724-Instruct-hf",
]

#: Set by hues.py (lineage: OLMo).
PALETTE = {"hue": 95}
QUIRKS = ["layernorm"]

BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "pre_norm_note": "A LayerNorm with no weight and no bias.",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} heads × {head_dim}, no bias",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "The MLP's input norm, named for its place after the attention's add; no weight and no bias.",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
    ],
}

STRIP = {
    "norm": "A LayerNorm with no weight and no bias.",
}

NOTES = """
## On 0424 and 0724 the queries, keys and values are clamped

Where the config sets `clip_qkv` (OLMo-7B-0424-hf, OLMo-1B-0724-hf and OLMo-7B-0724-hf), the
attention clamps the outputs of `q_proj`, `k_proj` and `v_proj` to `±clip_qkv` in place. A saved
`q_proj.output` holds the clamped tensor, and an edit written there is clamped after it; edit at
`attention_queries`, `attention_keys` and `attention_values`, which no clamp follows.
"""
