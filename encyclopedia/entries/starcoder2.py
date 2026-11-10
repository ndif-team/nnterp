"""StarCoder2: BigCode's code models, which load as Starcoder2ForCausalLM."""

MODEL_TYPE = "starcoder2"
TITLE = "StarCoder2"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "bigcode/starcoder2-3b"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-Starcoder2ForCausalLM"
CHECKPOINTS = [
    "bigcode/starcoder2-3b",
    "bigcode/starcoder2-7b",
    "bigcode/starcoder2-15b",
    "bigcode/starcoder2-15b-instruct-v0.1",
]

#: Set by hues.py (lineage: GPT-2).
PALETTE = {"hue": 41}
QUIRKS = ["layernorm", "qkv-bias", "sliding-window"]

#: What the visualization draws: Llama's sequential block, one LayerNorm before each sublayer.
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
            "detail": "{num_heads} heads, {num_kv_heads} kv; window {sliding_window}",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "The MLP's input norm, named for its place after the attention's add.",
            "contribution": "mlp_output",
            "detail": "GELU: {hidden_size} → {intermediate_size} → {hidden_size}, no gate",
        },
    ],
}

#: Notes on the model-level strip, by node: none, since nothing happens at the embeddings, the final norm or the head.
STRIP: dict[str, str] = {}

NOTES = """
## Common English words split into several tokens

The vocabulary is built for code, and `" Paris"` is two tokens, `" Par"` and `"is"`. Check that a
target word is one token before reading its probability, or pick another word.
"""
