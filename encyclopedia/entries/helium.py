"""Helium: Kyutai's Llama-shaped block whose rotary turns adjacent pairs of dimensions."""

MODEL_TYPE = "helium"
TITLE = "Helium"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "kyutai/helium-1-preview-2b"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-HeliumForCausalLM"
#: The one public checkpoint whose config says model_type helium; Helium 1 (kyutai/helium-1-2b and its
#: per-domain variants) is model_type llama and loads as the llama family.
CHECKPOINTS = ["kyutai/helium-1-preview-2b"]

#: Set by hues.py (lineage: Llama and its kin).
PALETTE = {"hue": 250}
QUIRKS = ["interleaved-rotary"]

#: Real values in the notes were measured on helium-1-preview-2b on a GPU, in bfloat16 unless the note says
#: float32; shapes, identities and the rotary recomputed exactly ran on the pinned tiny checkpoint.

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
            "detail": "{num_heads} heads × {head_dim}, interleaved rotary",
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

#: Notes on the model-level strip, by node.
STRIP: dict[str, str] = {}

NOTES = """
## The tokenizer prepends `<s>` though `bos_token` is `None`

Every encoded prompt starts with `<s>`, which takes most of the attention, while
`tokenizer.bos_token` is `None`. Code that checks `bos_token` before dropping position 0 keeps that
sink in its activation averages.
"""
