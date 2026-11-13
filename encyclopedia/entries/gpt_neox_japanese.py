"""GPT-NeoX-Japanese: ABEJA's Japanese 2.7B model, GPTNeoXJapaneseForCausalLM."""

MODEL_TYPE = "gpt_neox_japanese"
TITLE = "GPT-NeoX-Japanese"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "abeja/gpt-neox-japanese-2.7b"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-GPTNeoXJapaneseForCausalLM"
CHECKPOINTS = ["abeja/gpt-neox-japanese-2.7b"]

#: Set by hues.py (lineage: GPT-NeoX).
PALETTE = {"hue": 66}
QUIRKS = ["layernorm"]

#: What the visualization draws: a sequential block, input_layernorm before the attention and
#: post_attention_layernorm before the MLP; no post-norms.
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
            "detail": "{num_heads} heads × {head_dim}, rotary",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "The MLP's input norm, after the attention's add; unlike GPT-NeoX's, this block is sequential.",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
    ],
    "identity_note": "On the last block attention_output includes the dense_bias the block adds after the module, "
                     "so the plain sum is exact.",
}

#: Notes on the model-level strip, by node.
STRIP: dict[str, str] = {}

NOTES = """
## The prompt's first token is an attention sink

The tokenizer adds no beginning-of-sequence token, so the prompt's first token takes most of the
attention and its stream has a norm many times the other positions'. Leave position 0 out of activation
statistics and steering-vector norms.
"""
