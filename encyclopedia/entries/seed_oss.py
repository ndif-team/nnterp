"""Seed-OSS: ByteDance Seed's 36B dense models, which load as SeedOssForCausalLM."""

MODEL_TYPE = "seed_oss"
TITLE = "Seed-OSS"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "ByteDance-Seed/Seed-OSS-36B-Base"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-SeedOssForCausalLM"
CHECKPOINTS = [
    "ByteDance-Seed/Seed-OSS-36B-Base",
    "ByteDance-Seed/Seed-OSS-36B-Base-woSyn",
    "ByteDance-Seed/Seed-OSS-36B-Instruct",
]

#: Set by hues.py (lineage: Llama and its kin).
PALETTE = {"hue": 256}
QUIRKS = ["qkv-bias"]

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
            "detail": "{num_heads} heads, {num_kv_heads} kv; q, k, v biased",
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

NOTES = ""
