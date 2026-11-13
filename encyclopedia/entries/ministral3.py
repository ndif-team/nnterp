"""Ministral 3: the text model of the Ministral 3 vision-language checkpoints, Llama's block with position-scaled queries."""

MODEL_TYPE = "ministral3"
TITLE = "Ministral 3"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "mistralai/Ministral-3-3B-Base-2512"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "yujiepan/ministral-3-tiny-random"
#: Every checkpoint is a `mistral3` wrapper whose text_config is `ministral3`.
CHECKPOINTS = [
    "mistralai/Ministral-3-3B-Base-2512", "mistralai/Ministral-3-3B-Instruct-2512",
    "mistralai/Ministral-3-8B-Base-2512", "mistralai/Ministral-3-8B-Instruct-2512",
    "mistralai/Ministral-3-8B-Reasoning-2512",
    "mistralai/Ministral-3-14B-Base-2512", "mistralai/Ministral-3-14B-Instruct-2512",
]

#: The vision-language wrapper every checkpoint above is, keyed by its config.model_type. The vision encoder comes from
#: the checkpoint's vision_config.model_type (encyclopedia/vision/pixtral.py); what a config does not say is here.
WRAPPERS = {
    "mistral3": {
        "title": "Mistral 3",
        "pinned": "yujiepan/mistral-3-tiny-random",
        "projector": "an RMSNorm on each patch, patch_merger folding each 2 × 2 block of an image's patches into one "
                     "(concatenated, then merging_layer, a linear), then linear_1, GELU, linear_2",
        "projector_input": "`vision.tower_output[0]`",
        "quirks": ["pooled-projector"],
        "notes": """
## The projector reads the last block's output

A write to `vision.tower_output`, or to the last vision block, reaches the text model.
""",
    },
}

#: Set by hues.py (lineage: Llama and its kin).
PALETTE = {"hue": 269}
QUIRKS: list[str] = []

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
