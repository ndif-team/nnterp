"""Cohere 2: Command R7B and Command A (Cohere2ForCausalLM), and the Aya Vision and Command A Vision wrappers."""

MODEL_TYPE = "cohere2"
TITLE = "Command R7B / Command A"
SUBTITLE = (
    "Command R's parallel block on one bias-free LayerNorm, with three sliding-window blocks to each full one; "
    "only the sliding blocks apply rotary, and the logits are the head's output times logit_scale."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
#: Every CohereLabs repository is gated; this is an ungated copy of c4ai-command-r7b-12-2024 (its card names that base).
REFERENCE = "mlx-community/c4ai-command-r7b-12-2024-bf16"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "trl-internal-testing/tiny-Cohere2ForCausalLM"
CHECKPOINTS = [
    "CohereLabs/c4ai-command-r7b-12-2024",
    "CohereLabs/c4ai-command-r7b-arabic-02-2025",
    "CohereLabs/c4ai-command-a-03-2025",
    # Ungated copies of Command R7B and Command A.
    "mlx-community/c4ai-command-r7b-12-2024-bf16",
    "unsloth/c4ai-command-a-03-2025",
    # Vision-language wrappers around a Cohere 2 text model, each a key of WRAPPERS by its config's model_type.
    "CohereLabs/aya-vision-8b",
    "CohereLabs/command-a-vision-07-2025",
    "unsloth/aya-vision-8b",  # an ungated copy of aya-vision-8b
]

_PIXEL_SHUFFLE = """
## The projector pixel-shuffles 2 × 2 patches into one token

`downsample_factor` is 2: the projector lays a tile's patches out as their grid and stacks each
2 × 2 neighbourhood into one vector four times as wide, so a tile's `(image_size // patch_size) ** 2`
patches become a quarter as many image tokens. `linear_1` maps that vector to
`alignment_intermediate_size`, its two halves are combined as a SwiGLU (`silu(gate) * x`), and
`linear_2` maps the result to the text model's `hidden_size`. `model.projector.output` keeps the
grid, `[tiles, rows, columns, hidden_size]`; `vision.image_features` is it flattened over the tiles
and the grid, and those are the rows the scatter writes.
"""

_TILES = """
## An image is one tile or several, plus a thumbnail

The processor cuts an image into up to `max_patches` (12) tiles of `image_size` pixels, each a row
of the vision encoder's batch, and adds a thumbnail of the whole image as a last tile when there is
more than one. On the pinned tiny wrapper a square image is one tile and a 2 : 1 image three (two
tiles and the thumbnail); each tile gives 16 image tokens there (8 × 8 patches, shuffled to 4 × 4).
The tile markers the prompt carries around each run of image tokens are text tokens:
`vision.image_token_mask` marks only the image tokens.
"""

#: The vision-language wrappers of this family, keyed by the wrapper's config.model_type. The vision encoder comes
#: from the checkpoint's vision_config.model_type (encyclopedia/vision/); what a config does not say is here.
WRAPPERS = {
    "aya_vision": {
        "title": "Aya Vision",
        "pinned": "hf-tiny-v2/tiny-random-AyaVisionForConditionalGeneration",
        "projector": "multi_modal_projector: a 2 × 2 pixel shuffle of the last block's patches, a LayerNorm, "
                     "then linear_1, a SwiGLU and linear_2",
        "projector_input": "the last block's layer_output, before vision.norm",
        "quirks": ["pooled-projector", "tiled-images"],
        "notes": """
## The projector reads the last block, before `vision.norm`

`vision_feature_layer` is `-1` and `vision_feature_select_strategy` is `"full"`: the projector
receives the vision encoder's last hidden state, `vision.layers[-1].layer_output`, every patch kept.
`post_layernorm` (`vision.norm`) runs after it and `vision.tower_output` is computed and discarded:
on the pinned tiny wrapper zeroing `tower_output` leaves `vision.image_features` bit-identical, and
zeroing the last block's `layer_output` changes them.

```python
with model.trace(prompt, images=[image]):
    mask = model.vision.image_token_mask.save()
    last = model.vision.layers[-1].layer_output.save()
    fed = model.projector.input.save()
    features = model.vision.image_features.save()
    first = model.layers[0].input.save()

torch.equal(fed, last)                    # True: block -1, before vision.norm
torch.equal(first[mask], features)        # True: the scatter
```
""" + _PIXEL_SHUFFLE + """
On Aya Vision the shuffled vector is normed first: `model.projector.layernorm`, a LayerNorm with a
bias over the `4 × vision_hidden` channels, precedes `linear_1`.

## 169 image tokens per tile

Aya Vision 8B's vision encoder takes 364-pixel tiles in 14-pixel patches, 26 × 26 = 676 patches, and
the shuffle makes 13 × 13 = 169 image tokens per tile.
""" + _TILES,
    },
    "cohere2_vision": {
        "title": "Command A Vision",
        "pinned": "hf-tiny-v2/tiny-random-Cohere2VisionForConditionalGeneration",
        "projector": "multi_modal_projector: a 2 × 2 pixel shuffle of vision.tower_output, then linear_1, a SwiGLU "
                     "and linear_2, with no norm",
        "projector_input": "`vision.tower_output`",
        "quirks": ["pooled-projector", "tiled-images"],
        "notes": """
## The projector reads `tower_output`

The projector receives the vision encoder's `last_hidden_state`, the patches after `post_layernorm`,
so `model.projector.input == model.vision.tower_output` and a write to `tower_output` reaches the
text model. On the pinned tiny wrapper:

```python
with model.trace(prompt, images=[image]):
    mask = model.vision.image_token_mask.save()
    out = model.vision.tower_output.save()
    fed = model.projector.input.save()
    features = model.vision.image_features.save()
    first = model.layers[0].input.save()

torch.equal(fed, out)                     # True: the projector reads tower_output
torch.equal(first[mask], features)        # True: the scatter
```
""" + _PIXEL_SHUFFLE + """
Command A Vision's projector has no norm: the shuffled vector goes straight into `linear_1`.
""" + _TILES,
    },
}

PALETTE = {"hue": 188}
VLLM = True
QUIRKS = ["parallel-blocks", "layernorm", "sliding-window", "nope-blocks", "interleaved-rotary", "scaled-logits"]

#: What the visualization draws: one input_layernorm whose output both sublayers read (drawn once per branch,
#: one node), no post-norms; the two contributions join the stream in one add.
BLOCK = {
    "topology": "parallel",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "pre_norm_note": "One LayerNorm, drawn on both branches: the attention and the MLP read the same output "
                             "tensor, so an in-place edit of self_attn.input reaches the MLP too.",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} query, {num_kv_heads} key/value heads",
            "variants": {
                "sliding_attention": "{sliding_window}-token window, rotary",
                "full_attention": "full causal, no rotary",
            },
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "input_layernorm",
            "pre_norm_note": "One LayerNorm, drawn on both branches: the attention and the MLP read the same output "
                             "tensor, so an in-place edit of self_attn.input reaches the MLP too.",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
    ],
    "identity_note": "Exact in the block's own order, x + attn + mlp; another order differs by rounding.",
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "A plain lookup with no scale and no position embedding, so token_embeddings equals layers[0].input "
             "on text. Positions enter only through the rotary of the sliding-window blocks.",
    "norm": "Cohere2LayerNorm: subtracts the mean and multiplies by a weight, with no bias, computed in float32.",
    "head": "lm_head shares its weight with embed_tokens (tie_word_embeddings).",
    "logits": "logits = lm_head.output × logit_scale (0.25 on Command R7B and Command A). project_on_vocab multiplies too.",
}

NOTES = """
## One LayerNorm feeds both sublayers

```
h   = input_layernorm(x)
out = x + self_attn(h) + mlp(h)
```

The block is Command R's: `input_layernorm` is its only norm, a LayerNorm with a weight and no
bias, computed in float32. Its one output tensor is passed to the attention and then to the MLP,
so `self_attn.input` and `mlp.input` are the same tensor: an in-place edit of `self_attn.input`
also changes what the MLP reads, and assigning `input_layernorm.output` changes what both read.

## The contributions are the two modules' outputs

`attention_output` is `o_proj`'s output and `mlp_output` is `down_proj`'s. The block computes
`residual + attn + mlp`, so that sum reproduces `layer_output` bit for bit; summed in another
order it differs by rounding. Zeroing `attention_output` leaves that block's `mlp.input` unchanged.

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

assert torch.equal(x + attn + mlp, out)           # the block's own order
```

## Three sliding blocks to one full block, and only the sliding ones turn

`config.layer_types` is `full_attention` on every fourth block (3, 7, ..., 31 on Command R7B's 32)
and `sliding_attention` elsewhere. A sliding block attends over a window of 4096 tokens; on a
prompt no longer than the window both masks are the same causal mask. A block's kind is
`model.layers[i].self_attn._module.sliding_window` (the window, or `None` on a full block).

Only the sliding blocks apply the rotary. A full block has no position encoding: its
`attention_queries` and `attention_keys` are `q_proj`'s and `k_proj`'s outputs split into heads,
and only the causal mask orders the tokens there. On a sliding block they are read after the
rotary, which turns adjacent pairs of dimensions (`2i` with `2i + 1`) at base 50,000.

```python
full = model.config.layer_types.index("full_attention")
with model.trace(prompt):
    k_proj = model.layers[full].self_attn.k_proj.output.save()
with model.trace(prompt):
    keys = model.layers[full].self_attn.attention_keys.save()

heads = k_proj.view(*k_proj.shape[:2], -1, model.head_dim).transpose(1, 2)
assert torch.equal(heads, keys)                   # no rotary on a full block
```

## Attention

Command R7B has 32 query heads and 8 key/value heads, Command A 96 and 8, so an edit to a key or
value head reaches 4 or 12 query heads; `attention_keys` and `attention_values` are served before
`repeat_kv`. The query scale is `head_dim ** -0.5` (128). Both kinds of block make the same
attention-interface call, so every interior value resolves on every block, under
`attn_implementation="eager"` (the values marked `⚠`). There are no query or key norms.

## The logits are the head's output times `logit_scale`

`lm_head` is the embedding matrix (`tie_word_embeddings`), and the model multiplies its output by
`config.logit_scale`, 0.25 on Command R7B and Command A. `logits` is the scaled tensor and
`lm_head.output` the unscaled one. The family imports Command R's `project_on_vocab` (the final
norm, `lm_head`, the scale), so a logit lens on the last block equals `logits` exactly.

```python
with model.trace(prompt):
    resid = model.layers[-1].layer_output.save()
    raw = model.lm_head.output.save()
    logits = model.logits.save()

assert torch.equal(logits, raw * model.config.logit_scale)
assert torch.equal(model.project_on_vocab(resid), logits)
```

## Aya Vision and Command A Vision wrap this text model

`aya_vision` and `cohere2_vision` checkpoints load with `task="image-text-to-text"`; nnterp picks this
family from `config.text_config` and binds `embed_tokens`, `layers` and `norm` under
`model.language_model`, with `lm_head` at the root. The vision encoder is SigLIP at
`model.vision_tower`. The image features replace the image tokens' embeddings after
`embed_tokens`, so at those positions `token_embeddings` is the placeholder token's embedding and
`layers[0].input` is the image; at text positions the two are equal. `embed_tokens` runs before
the vision encoder, so read `token_embeddings` before any `vision` value in a trace.
"""
