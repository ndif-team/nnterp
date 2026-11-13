"""Qwen2-VL: the text model of the Qwen2-VL checkpoints, Qwen2's block under multimodal rotary embeddings."""

MODEL_TYPE = "qwen2_vl_text"
TITLE = "Qwen2-VL"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "Qwen/Qwen2-VL-2B-Instruct"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "yujiepan/qwen2-vl-tiny-random"
#: Every checkpoint is the vision-language wrapper (model_type qwen2_vl): transformers has no causal-LM class for it.
CHECKPOINTS = [
    "Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-2B",
    "Qwen/Qwen2-VL-7B-Instruct", "Qwen/Qwen2-VL-7B",
    "Qwen/Qwen2-VL-72B-Instruct", "Qwen/Qwen2-VL-72B",
]

#: The vision-language wrapper every checkpoint above is, keyed by its config.model_type. The vision encoder comes from
#: the checkpoint's vision_config.model_type (encyclopedia/vision/qwen_vit.py); what a config does not say is here.
WRAPPERS = {
    "qwen2_vl": {
        "title": "Qwen2-VL",
        "pinned": "yujiepan/qwen2-vl-tiny-random",
        "projector": "merger, inside the vision encoder: a LayerNorm (ln_q) on each patch, then an MLP (Linear, GELU, "
                     "Linear) over each 2 × 2 block of patches concatenated, one image token per block",
        "projector_input": "the merger's input: the last block's output",
    },
}

#: Set by hues.py (lineage: Qwen).
PALETTE = {"hue": 288}
QUIRKS = ["qkv-bias", "multimodal-rotary"]

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
            "detail": "{num_heads} heads, {num_kv_heads} kv; M-RoPE, qkv bias",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}, no biases",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP: dict[str, str] = {}

NOTES = ""
