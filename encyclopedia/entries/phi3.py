"""Phi-3, Phi-3.5 and Phi-4: Microsoft's models that load as Phi3ForCausalLM."""

MODEL_TYPE = "phi3"
TITLE = "Phi-3 / Phi-3.5 / Phi-4"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "microsoft/Phi-3-mini-4k-instruct"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "katuni4ka/tiny-random-phi3"
CHECKPOINTS = [
    "microsoft/Phi-3-mini-4k-instruct", "microsoft/Phi-3-mini-128k-instruct",
    "microsoft/Phi-3-medium-4k-instruct", "microsoft/Phi-3-medium-128k-instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "microsoft/Phi-4-mini-instruct", "microsoft/Phi-4-mini-reasoning",
    "microsoft/phi-4", "microsoft/Phi-4-reasoning", "microsoft/Phi-4-reasoning-plus",
]

#: Set by hues.py (lineage: Phi).
PALETTE = {"hue": 140}
QUIRKS = [
    # The 128k checkpoints' windows reach past their context and never mask; phi-4 sets none.
    {"slug": "sliding-window",
     "when": lambda config: config.sliding_window is not None and config.sliding_window < config.max_position_embeddings},
    # Only Phi-4-mini sets a factor below 1.
    {"slug": "partial-rotary", "when": lambda config: config.rope_parameters.get("partial_rotary_factor", 1.0) < 1.0},
]

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
            "detail": "{num_heads} heads, {num_kv_heads} kv, fused qkv",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "The MLP's input norm, named for its place after the attention's add.",
            "contribution": "mlp_output",
            "detail": "fused gate_up: {hidden_size} → 2 × {intermediate_size}, {hidden_act}",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP: dict[str, str] = {}

NOTES = """
## On Phi-3 and Phi-3.5 `" Paris"` is two tokens

Their SentencePiece tokenizer splits a word after a space as `' Paris'` → `['▁', '▁Paris']`, so the
first token of `" Paris"` is a bare `▁`. Encode `"Paris"` to get the word's one token, `▁Paris`, as
the answer token for a logit metric.
"""
