"""The Qwen ViT: Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3.5 and their mixture-of-experts lines. Packed, its merger inside."""

TITLE = "Qwen ViT"
#: Each line's vision config type; the tiny Qwen2-VL and Qwen2.5-VL checkpoints the suites pin report the wrapper's
#: type (`qwen2_vl`, `qwen2_5_vl`) in their vision_config.
VISION_CONFIG_TYPES = [
    "qwen2_vl_vision", "qwen2_vl",
    "qwen2_5_vl_vision", "qwen2_5_vl",
    "qwen3_vl_vision", "qwen3_vl_moe_vision",
    "qwen3_5_vision", "qwen3_5_moe_vision",
]
MODULE_CLASSES = [
    "Qwen2VisionTransformerPretrainedModel", "Qwen2_5_VisionTransformerPretrainedModel",
    "Qwen3VLVisionModel", "Qwen3VLMoeVisionModel",
    "Qwen3_5VisionModel", "Qwen3_5MoeVisionModel",
]

BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "pre_norm_note": "The native norm1: a LayerNorm, an RMSNorm on Qwen2.5-VL.",
            "contribution": "attention_output",
            # One interface call per image (per window on Qwen2.5-VL), so no pattern: scores and probabilities are Unavailable.
            "interior": ["attention_queries", "attention_keys", "attention_values", "attention_head_outputs"],
            "detail": "{num_heads} heads × {head_dim}, per image",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "The native norm2: a LayerNorm, an RMSNorm on Qwen2.5-VL.",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
    ],
}

ROWS = ("One row, packed: every image's patches in one row, `[1, patches, vision_hidden]`, the processor's "
        "`image_grid_thw` splitting it per image; each image's patches in merge-block order (each 2 × 2 block the "
        "merger folds is consecutive), and in window order inside the vision encoder on Qwen2.5-VL.")
MASKING = ("The attention runs once per image, over that image's patches, so no patch attends to another image's; "
           "on Qwen2.5-VL's windowed blocks once per window.")
POSITIONS = ("A 2D rotary embedding (row and column of the patch) turns the queries and keys inside each attention. "
             "Qwen3-VL and Qwen3.5 also add a learned `pos_embed`, resampled to each image's grid, after `patch_embed`.")
NORM = ("None after the last block: the merger norms each patch as it reads it, so there is no `vision.norm`.")
QUIRKS = ["packed-tower", "variable-resolution", "pooled-projector"]

NOTES = """
## Every image in one row

The vision encoder runs on `[patches, vision_hidden]`, the patches of every image in the invoke
concatenated. Its stream values (`patch_embeddings`, every block's `layer_output`,
`attention_output` and `mlp_output`, `tower_output`) are served `[1, patches, vision_hidden]`, a view with a
leading images axis of 1, so in-place edits land; `vision.layers[i].input` and `.output` stay the
native `[patches, vision_hidden]`, and the block identity holds across the two by broadcasting. The
processor's `image_grid_thw` (`[t, h, w]` per image, in patches; an image is two identical frames, so
`t` is 1) splits the row: image `j` has `t * h * w` patches.

```python
inputs = model.processor(text=prompt, images=[red, wide], return_tensors="pt")
sizes = inputs["image_grid_thw"].prod(-1).tolist()   # patches per image
with model.trace(prompt, images=[red, wide]):
    patches = model.vision.layers[1].layer_output.save()
per_image = patches[0].split(sizes)                  # one [t*h*w, vision_hidden] each
```

Inside an image the patches are in merge-block order: the `spatial_merge_size` ×
`spatial_merge_size` (2 × 2) block of neighbours the merger folds into one image token is four
consecutive rows, blocks in raster order. Index `4 * b` to `4 * b + 3` is block `b`, not a raster row.

## The attention runs once per image, so there is no pattern

The attention computes queries, keys and values for the whole row, then calls the attention
function once per image (once per window on Qwen2.5-VL's windowed blocks) and concatenates the
outputs. `attention_queries`, `attention_keys` and `attention_values` are served whole, before the
split, `[1, heads, patches, head_dim]`, the queries and keys after the rotary embedding;
`attention_head_outputs` is the calls' outputs concatenated back, `[1, patches, heads, head_dim]`
(under any implementation but flash, which runs one call over `cu_seqlens`). `attention_scores` and
`attention_probabilities` are `Unavailable` under every implementation, eager included: "no one
tensor is the block's pattern". The split points are the attention's `cu_seqlens` argument,
`self_attn.inputs[1]["cu_seqlens"]`, so a pattern per image is a softmax over each slice:

```python
attn = model.vision.layers[1].self_attn
with model.trace(prompt, images=[red, wide]):
    cu = attn.inputs[1]["cu_seqlens"].save()        # read first: the module's input
    q = attn.attention_queries.save()
    k = attn.attention_keys.save()

patterns = []
for a, b in zip(cu[:-1].tolist(), cu[1:].tolist()):
    scores = q[0, :, a:b] @ k[0, :, a:b].transpose(-1, -2) * model.vision.head_dim ** -0.5
    patterns.append(scores.softmax(-1))             # [heads, patches_j, patches_j]
```

Each pattern times that image's values reproduces `attention_head_outputs` exactly (checked on
Qwen3.5's pinned tiny checkpoint).

## The merger is inside the vision encoder

`model.projector` is the vision encoder's own `merger`: it norms each patch, concatenates each 2 × 2
block into one `4 * vision_hidden` vector and runs a two-layer MLP to the text model's width, so
image tokens per image are `t * h * w / 4`. It reads the last block's stream with no norm between
(`projector.input == tower_output[0]`, the native `[patches, vision_hidden]`). Because the merger
runs inside the vision encoder's forward, `vision.tower_output` (the encoder's output) is reached after
it: read `model.projector.input` and `.output` before `vision.tower_output` in a trace.

## Sizes, and no `image_size`

`model.vision` reads its sizes off the vision config: `hidden_size` is the encoder's width
(`embed_dim` on Qwen2-VL, whose config's `hidden_size` is the merger's output width),
`num_heads`, `intermediate_size` (`embed_dim * mlp_ratio` on Qwen2-VL), `patch_size`, and two of
its own, `spatial_merge_size` (2 on every line) and `window_size` (Qwen2.5-VL's window side in
pixels, 112; `None` on the others). `image_size` raises `Unavailable`: the processor cuts each
image into its own grid.

## The variants

| line | block norms, merger norm | MLP | beyond the shared encoder |
|---|---|---|---|
| Qwen2-VL | LayerNorm | `fc1`, QuickGELU, `fc2` | none |
| Qwen2.5-VL | RMSNorm | SwiGLU, biased | windows of 112 pixels except on `fullatt_block_indexes` |
| Qwen3-VL, -MoE | LayerNorm | `linear_fc1`, GELU (tanh), `linear_fc2` | learned `pos_embed`; DeepStack taps |
| Qwen3.5, -MoE | LayerNorm | `linear_fc1`, GELU (tanh), `linear_fc2` | learned `pos_embed` |

On Qwen2.5-VL the encoder permutes the patches into windows at entry, so its blocks' values,
`tower_output` and `projector.output` are in window order, and it restores the merge-block order
after the merger: `vision.image_features` is in scatter order. On Qwen3-VL and Qwen3.5 `vision.layers[0].input`
is `patch_embeddings` plus the resampled `pos_embed`. On Qwen2-VL `vision.layers[0].input` is
`patch_embeddings`; on Qwen2.5-VL it is `patch_embeddings` permuted into window order; on both, positions
enter only through the rotary embedding.
"""
