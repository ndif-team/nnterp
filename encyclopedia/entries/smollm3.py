"""SmolLM3: Hugging Face's 3B models, which load as SmolLM3ForCausalLM."""

MODEL_TYPE = "smollm3"
TITLE = "SmolLM3"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "HuggingFaceTB/SmolLM3-3B-Base"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "yujiepan/smollm3-tiny-random"
CHECKPOINTS = [
    "HuggingFaceTB/SmolLM3-3B-Base",
    "HuggingFaceTB/SmolLM3-3B",
]

#: Set by hues.py (lineage: Llama and its kin).
PALETTE = {"hue": 247}
QUIRKS = ["nope-blocks"]

#: What the visualization draws: Llama's sequential block, one RMSNorm before each sublayer.
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
            "detail": "{num_heads} heads, {num_kv_heads} kv; NoPE 1 in {no_rope_layer_interval}",
            "variants": {
                "full_attention": "{num_heads} heads, {num_kv_heads} kv; NoPE 1 in {no_rope_layer_interval}",
                "sliding_attention": "{num_heads} heads, {num_kv_heads} kv; sliding window",
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
## Every fourth block has no rotary, and `layer_types` does not say so

`config.no_rope_layers` is 0 on blocks 3, 7, 11, ..., 35, which apply no rotary embedding, while
`layer_types` lists every block as `full_attention`. Pick the NoPE blocks from `no_rope_layers`.
"""
