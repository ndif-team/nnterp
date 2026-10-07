"""SigLIP's vision encoder, and the SigLIP-shaped ViT of Idefics 3 and SmolVLM: patches only, a final norm over them."""

TITLE = "SigLIP"
#: Idefics 3 and SmolVLM report their own types for the same vision encoder shape.
VISION_CONFIG_TYPES = ["siglip_vision_model", "idefics3", "idefics3_vision", "smolvlm_vision"]
MODULE_CLASSES = ["SiglipVisionModel", "Idefics3VisionTransformer", "SmolVLMVisionTransformer"]

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
            "detail": "{num_heads} heads × {head_dim}, no causal mask",
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

ROWS = "One row per image (per tile on Idefics 3 and SmolVLM); patches in raster order."
MASKING = "No causal mask: a patch attends to every patch of its row."
POSITIONS = ("Learned absolute position embeddings are added to the patch embeddings; no class token and no norm "
             "before block 0.")
NORM = "`post_layernorm` norms every patch after the last block: it is `vision.norm`, and `vision.tower_output` is its output."
QUIRKS: list[str] = []

NOTES = """
## The patch axis is the patch grid

No class token: `vision.patch_embeddings`, every block's `layer_output` and `vision.tower_output` have
the same length, `(image_size // patch_size) ** 2` patches per image or tile, in raster order. Row `r`,
column `c` of the grid is index `r * (image_size // patch_size) + c` on every value of the vision encoder.

## The final norm is `vision.norm`

`post_layernorm` norms every patch after the last block, so
`vision.tower_output == vision.norm(vision.layers[-1].layer_output)`: a write to the last block's
`layer_output` reaches `tower_output` through the norm.
"""
