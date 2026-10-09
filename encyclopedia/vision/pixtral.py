"""Pixtral's vision encoder: every image's patches packed in one row under a block-diagonal mask, 2D rotary positions, no final norm."""

TITLE = "Pixtral"
VISION_CONFIG_TYPES = ["pixtral"]
MODULE_CLASSES = ["PixtralVisionModel"]

BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "pre_norm_note": "The native attention_norm, an RMSNorm.",
            "contribution": "attention_output",
            # One interface call over the packed row, under a block-diagonal mask: the interior is whole.
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} heads × {head_dim}, block-diagonal",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "The native ffn_norm, an RMSNorm: the MLP's input norm.",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
    ],
}

ROWS = ("One row, packed: every image's patches, image after image, `[1, patches, vision_hidden]`; image `j` has "
        "`(height // patch_size) * (width // patch_size)` patches in raster order, its `image_sizes` entry from the "
        "processor. `vision.patch_embeddings` is the packed row entering `ln_pre`; the convolution's own output, "
        "`patch_embed.output`, is the padded grid.")
MASKING = ("Block-diagonal: a patch attends to every patch of its own image and to none of another's, so the pattern "
           "is zero between images.")
POSITIONS = ("A 2D rotary embedding (the patch's row and column in its image) turns the queries and keys inside each "
             "attention; nothing is added to the stream. `ln_pre` norms the packed row before block 0.")
NORM = "None after the last block, so there is no `vision.norm`, and `vision.tower_output` is the last block's `layer_output`."
QUIRKS = ["packed-tower", "variable-resolution"]

NOTES = """
## Two images are one row

The convolution runs on the processor's batch, padded to its largest image, and each image's grid
is cropped back to its own size, flattened and concatenated: `vision.patch_embeddings` is
`[1, patches, vision_hidden]`, the patches of every image in the invoke, image after image, each in
raster order. `vision.patch_embed.output` is the padded grid, `[images, vision_hidden, rows, columns]`,
and is not the stream. Every block value is served on the packed row, so the processor's
`image_sizes` split it:

```python
inputs = model.processor(text=prompt, images=[square, wide], return_tensors="pt")
side = model.vision.patch_size
sizes = [h // side * (w // side) for h, w in inputs["image_sizes"].tolist()]
with model.trace(prompt, images=[square, wide]):
    patches = model.vision.layers[1].layer_output.save()
per_image = patches[0].split(sizes)                 # one [patches_j, vision_hidden] each
```

On the pinned Mistral 3 tiny checkpoint a 36 × 36 and a 24 × 36 image at 6-pixel patches are
36 and 24 patches: `patch_embed.output` is `[2, 32, 6, 6]` and `patch_embeddings` `[1, 60, 32]`.

## `ln_pre` sits between the patches and block 0

`ln_pre`, an RMSNorm, norms the packed row before the first block, so `vision.layers[0].input` is
`ln_pre`'s output and not `vision.patch_embeddings`. No position embedding is added: positions
enter only as the 2D rotary inside each attention, by the patch's row and column within its own
image.

## The attention runs once over the row, under a block-diagonal mask

Each block calls the attention function once on the whole row with a mask that is zero inside an
image's square and the dtype's minimum between images. So the interior is whole: under
`attn_implementation="eager"`, `attention_probabilities` is `[1, heads, patches, patches]`, each row
summing to one over its own image's patches and exactly zero on the other images'.

```python
with model.trace(prompt, images=[square, wide]):
    pattern = model.vision.layers[0].self_attn.attention_probabilities.save()

a = sizes[0]
pattern[..., :a, a:].abs().max(), pattern[..., a:, :a].abs().max()   # 0.0, 0.0
```

## No final norm, and no `image_size`

The last block's stream leaves the vision encoder as it is: `vision.tower_output` equals
`vision.layers[-1].layer_output`, and `model.vision` has no `norm`. `vision.image_size` raises
`Unavailable`: the processor resizes each image to fit its own grid, and the config's
`image_size` is the longest side it resizes to, not a size every image has.
"""
