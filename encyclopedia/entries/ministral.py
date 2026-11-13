"""Ministral: Llama's block with full-attention and sliding-window blocks interleaved by config.layer_types."""

MODEL_TYPE = "ministral"
TITLE = "Ministral"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
#: Its config.json says `mistral`; AutoConfig reads a `mistral` config with `layer_types` as `ministral`.
REFERENCE = "mistralai/Ministral-8B-Instruct-2410"
#: The tiny checkpoint the test suite builds the page from (its test rewrites it into a local copy with a tokenizer).
PINNED = "hf-tiny-v2/tiny-random-MinistralForCausalLM"
CHECKPOINTS = ["mistralai/Ministral-8B-Instruct-2410"]

#: Set by hues.py (lineage: Llama and its kin).
PALETTE = {"hue": 266}
QUIRKS = ["sliding-window"]

#: What the visualization draws: the sublayers in forward order, each with its norms,
#: the standard value that is its contribution, and the values read inside it.
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
            "detail": "{num_heads} query, {num_kv_heads} key/value heads",
            "variants": {
                "full_attention": "full causal attention",
                "sliding_attention": "sliding window of {sliding_window} tokens",
            },
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "The MLP's input norm, named for its place after the attention's add.",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
    ],
}

#: Notes on the model-level strip, by node: none, since nothing happens at the embeddings, the final norm or the head.
STRIP: dict[str, str] = {}

NOTES = """
## The window is as long as the context

On Ministral-8B-Instruct-2410 `sliding_window` is 32768, the config's `max_position_embeddings`, so up
to that length a sliding block's mask is the causal mask, and the sliding and full blocks attend
alike.
"""
