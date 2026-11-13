"""Mistral: Llama's block, with a sliding window on the first 7B and two vision encoders on its wrappers."""

MODEL_TYPE = "mistral"
TITLE = "Mistral / Mistral Nemo / Mistral Small"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "mistralai/Mistral-7B-v0.1"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-MistralForCausalLM"
CHECKPOINTS = [
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-v0.3", "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-Nemo-Base-2407",
    # Vision-language wrappers around a Mistral text model, each a key of WRAPPERS by its config's model_type.
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503",  # mistral3, Pixtral
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",  # mistral3, Pixtral
    "mistral-community/pixtral-12b",                  # llava, Pixtral
    "llava-hf/llava-v1.6-mistral-7b-hf",              # llava_next, CLIP
    "llava-hf/bakLlava-v1-hf",                        # llava, CLIP
]

#: The vision-language wrappers of this family, keyed by the wrapper's config.model_type. The vision encoder comes
#: from the checkpoint's vision_config.model_type (encyclopedia/vision/pixtral.py, clip.py); what a config does not
#: say is here. The `llava` key holds two checkpoints on two vision encoders (Pixtral-12B and BakLLaVA): its
#: per_checkpoint gives each its own projector reading and notes.
WRAPPERS = {
    "mistral3": {
        "title": "Mistral 3",
        "pinned": "hf-tiny-v2/tiny-random-Mistral3ForConditionalGeneration",
        "projector": "an RMSNorm on each patch, patch_merger folding each 2 × 2 block of an image's patches into one "
                     "(concatenated, then merging_layer, a linear), then linear_1, GELU, linear_2",
        "projector_input": "`vision.tower_output[0]`",
        "quirks": ["pooled-projector"],
        "notes": """
## The projector reads the last block's output

A write to `vision.tower_output`, or to the last vision block, reaches the text model.
""",
    },
    "llava": {
        "title": "Llava",
        "pinned": "mistral-community/pixtral-12b",
        "projector": "a two-layer MLP (linear_1, GELU, linear_2), one image token per patch, over the block "
                     "vision_feature_layer names",
        "projector_input": "the block `vision_feature_layer` names, sliced by `vision_feature_select_strategy`",
        "notes": "",
        # Two checkpoints of this wrapper class on two vision encoders, each reading another block.
        "per_checkpoint": {
            "mistral-community/pixtral-12b": {
                "title": "Llava (Pixtral-12B)",
                "projector": "a two-layer MLP (linear_1, GELU, linear_2), one image token per patch, over the last "
                             "block (vision_feature_layer -1, \"full\")",
                "projector_input": "`vision.tower_output`",
                "notes": """
## The projector reads `vision.tower_output`

A write to `vision.tower_output`, or to the last vision block, reaches the text model.
""",
            },
            "llava-hf/bakLlava-v1-hf": {
                "title": "Llava (BakLLaVA)",
                "projector": "a two-layer MLP (linear_1, GELU, linear_2), one image token per patch, over block -2 "
                             "without the CLS token (vision_feature_layer -2, \"default\")",
                "projector_input": "`vision.layers[-2].layer_output[:, 1:]`",
                "notes": """
## The projector reads block -2 without its CLS token

The last block and `vision.tower_output` are computed and discarded, so a write to them does not
reach the text model.
""",
            },
        },
    },
    "llava_next": {
        "title": "LLaVA-NeXT",
        "pinned": "trl-internal-testing/tiny-LlavaNextForConditionalGeneration",
        "projector": "a two-layer MLP over each crop's block -2 patches, the CLS token dropped; the wrapper unpads its "
                     "output and adds newline tokens",
        "projector_input": "each crop's `vision.layers[-2].layer_output[:, 1:]`",
        "quirks": ["tiled-images", "unpadded-features"],
        "notes": """
## The projector reads block -2 without its CLS token

The last block and `vision.tower_output` are computed and discarded, so a write to them does not
reach the text model.
""",
    },
}

#: Set by hues.py (lineage: Llama and its kin).
PALETTE = {"hue": 259}
QUIRKS = [
    # Mistral-7B-v0.1 and BakLLaVA set sliding_window; v0.3, Nemo, Small and Pixtral-12B leave it null
    {"slug": "sliding-window", "when": lambda config: config.sliding_window is not None},
]

#: What the visualization draws: the sublayers in forward order, each with its norms,
#: the standard value that is its contribution, and the values read inside it.
#: ``detail`` is formatted with the sizes and the config keys.
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
