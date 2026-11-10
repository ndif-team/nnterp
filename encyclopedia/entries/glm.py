"""GLM-4-9B in transformers' own format (the ``-hf`` repositories) and GLM-Edge: ``GlmForCausalLM``."""

MODEL_TYPE = "glm"
TITLE = "GLM-4 / GLM-Edge"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "zai-org/glm-4-9b-hf"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-GlmForCausalLM"
CHECKPOINTS = [
    "zai-org/glm-4-9b-hf", "zai-org/glm-4-9b-chat-hf", "zai-org/glm-4-9b-chat-1m-hf",
    "zai-org/glm-edge-1.5b-chat", "zai-org/glm-edge-4b-chat",
]

#: Set by hues.py (lineage: GLM).
PALETTE = {"hue": 116}
QUIRKS = [
    {"slug": "qkv-bias", "when": lambda config: config.attention_bias},  # GLM-4-9B, not GLM-Edge
    {"slug": "partial-rotary", "when": lambda config: config.rope_parameters.get("partial_rotary_factor", 1.0) < 1.0},  # GLM-4-9B
    "interleaved-rotary",
]

#: Every claim in the notes was checked on the pinned tiny checkpoint (shapes, identities, the rotary
#: recomputed exactly) or read off the listed checkpoints' configs and tokenizers; no real weights were run.

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
            "detail": "{num_heads} heads, {num_kv_heads} kv, × {head_dim}",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "detail": "fused gate_up_proj: {hidden_size} → 2 × {intermediate_size} → {hidden_size}, {hidden_act}",
            "pre_norm_note": "The MLP's input norm, as on Llama.",
        },
    ],
}

STRIP: dict[str, str] = {}

NOTES = """
## GLM-4-9B's tokenizer prepends `[gMASK]<sop>`

Every encoded string starts with these two tokens, so the prompt's first token is at position 2. The
chat template writes them too, so tokenize a templated prompt with `add_special_tokens=False` or the
prefix appears twice. GLM-Edge's tokenizer prepends nothing.
"""
