"""Arcee AFM: Llama's block with a two-projection MLP around a squared ReLU."""

MODEL_TYPE = "arcee"
TITLE = "Arcee AFM"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "arcee-ai/AFM-4.5B-Base"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-ArceeForCausalLM"
CHECKPOINTS = [
    "arcee-ai/AFM-4.5B-Base", "arcee-ai/AFM-4.5B",
    "arcee-ai/AFM-4.5B-Preview", "arcee-ai/AFM-4.5B-Base-Pre-Anneal",
]

#: Set by hues.py (lineage: Llama and its kin).
PALETTE = {"hue": 235}
QUIRKS = ["squared-relu"]

#: Every checkpoint is 4.5B parameters, above the size a real run here allows, so no real weights were run:
#: shapes, identities and the activation ran on the pinned tiny checkpoint (whose hidden_act is gelu; the
#: activation note loads it with hidden_act="relu2"), the activation class on a meta build of AFM-4.5B-Base,
#: and the rest is read off the configs and tokenizers.

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
            "detail": "{num_heads}/{num_kv_heads} heads × {head_dim}",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "The MLP's input norm, named for its place after the attention's add.",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}, no gate",
        },
    ],
}

#: Notes on the model-level strip, by node: none, since nothing happens at the embeddings, the final norm or the head.
STRIP: dict[str, str] = {}

NOTES = ""
