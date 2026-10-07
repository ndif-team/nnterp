"""CLIP's vision tower: a class token before the patches, a norm before block 0, none over the patches after the last."""

TITLE = "CLIP"
VISION_CONFIG_TYPES = ["clip_vision_model"]
MODULE_CLASSES = ["CLIPVisionModel"]

BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "pre_norm_note": "The native layer_norm1, a LayerNorm.",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} heads × {head_dim}, no mask",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "The native layer_norm2, a LayerNorm.",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
    ],
}

ROWS = "One row per image (per crop on LLaVA-NeXT); the CLS token first, then the patches in raster order."
MASKING = "No mask: every position, the CLS token included, attends to every position of its row."
POSITIONS = ("Learned absolute position embeddings, one per position, the class embedding's included, are added after "
             "the patch embedding, and `pre_layrnorm` norms the result: `vision.layers[0].input` is `pre_layrnorm`'s output.")
NORM = ("None over the patches: `post_layernorm` norms the pooled CLS token only, so there is no `vision.norm`, and "
        "`vision.tower_output` is the last block's `layer_output`.")
QUIRKS = ["cls-token"]

NOTES = """
## The CLS token comes first

`vision.patch_embeddings` is the patch grid alone, `[images, grid, vision_hidden]`. The class
embedding is prepended after it, so from `vision.layers[0].input` on the stream is one row longer:
`[:, 0]` is the CLS token and `[:, 1:]` the patches in raster order. Slice `[:, 1:]` before mapping a
row to a patch.

## A norm before block 0, none after the last

The position embeddings are added to the CLS token and the patches, and `pre_layrnorm` norms the
result: `vision.layers[0].input` is `pre_layrnorm`'s output, not `patch_embeddings` plus positions.
After the last block, `post_layernorm` norms the pooled CLS token only, never the patches, so it is
not `vision.norm`, and `vision.tower_output` equals `vision.layers[-1].layer_output`.
"""
