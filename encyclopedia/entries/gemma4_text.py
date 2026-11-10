"""Gemma 4, text: a sandwich block with per-layer embeddings, a learned scale on each block's sum, and borrowed keys and values."""

MODEL_TYPE = "gemma4_text"
TITLE = "Gemma 4"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "google/gemma-4-E2B"
#: The tiny checkpoint the test suite builds the page from: the suite tiles its per-layer table to the
#: full vocabulary in a local copy (``_ple_checkpoint``), and the page is built from that copy.
PINNED = "hf-tiny-v2/tiny-random-Gemma4ForCausalLM"
#: Every checkpoint is a vision-language wrapper (model_type gemma4, a key of WRAPPERS).
CHECKPOINTS = [
    "google/gemma-4-E2B", "google/gemma-4-E2B-it",
    "google/gemma-4-E4B", "google/gemma-4-E4B-it",
    "google/gemma-4-26B-A4B", "google/gemma-4-26B-A4B-it",
    "google/gemma-4-31B", "google/gemma-4-31B-it",
]

#: The vision-language wrapper of this family, keyed by the wrapper's config.model_type. Its vision encoder is hosted
#: by this family alone, so it is inline (``tower``) rather than a module under encyclopedia/vision/.
WRAPPERS = {
    "gemma4": {
        "title": "Gemma 4",
        "pinned": "yujiepan/gemma-4-e-tiny-random",
        "projector": "embed_vision: an RMS norm without gain (embedding_pre_projection_norm) and a linear "
                     "(embedding_projection) over the soft tokens the vision encoder's pooler returns",
        "projector_input": "the pooler's soft tokens, after `vision.tower_output`",
        "tower": {
            "TITLE": "Gemma 4 ViT",
            "VISION_CONFIG_TYPES": ["gemma4_vision"],
            "MODULE_CLASSES": ["Gemma4VisionModel"],
            "BLOCK": {
                "topology": "sequential",
                "sublayers": [
                    {
                        "host": "self_attn",
                        "kind": "attention",
                        "label": "Attention",
                        "pre_norm": "input_layernorm",
                        "post_norm": "post_attention_layernorm",
                        "post_norm_note": "This norm follows the attention, as on the text block: its output is attention_output, "
                                         "and layer_mid is read at pre_feedforward_layernorm's input.",
                        "contribution": "attention_output",
                        "interior": [
                            "attention_queries", "attention_keys", "attention_values",
                            "attention_scores", "attention_probabilities", "attention_head_outputs",
                        ],
                        "detail": "{num_heads} heads × {head_dim}, 2D rotary",
                    },
                    {
                        "host": "mlp",
                        "kind": "mlp",
                        "label": "MLP",
                        "pre_norm": "pre_feedforward_layernorm",
                        "post_norm": "post_feedforward_layernorm",
                        "contribution": "mlp_output",
                        "detail": "GeGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_activation}",
                    },
                ],
            },
            "ROWS": ("One row per image: its patches, then padding up to "
                     "`max_soft_tokens * pooling_kernel_size**2` rows (2520 by default); the processor's "
                     "`image_position_ids` is `(-1, -1)` on the padded rows."),
            "MASKING": "No causal mask: a row attends to every patch of its image; the padded rows are masked as keys.",
            "POSITIONS": ("`patch_embeddings` is `patch_embedder.input_proj`'s output; `patch_embedder` then adds a learned "
                          "x and a learned y embedding per patch (zero on the padded rows), and every attention rotates "
                          "queries and keys by the patch's 2D position."),
            "NORM": ("None: `vision.tower_output` is the encoder's output, padded rows included, and the vision "
                     "encoder's `pooler` follows it, outside the blocks."),
            "QUIRKS": ["variable-resolution", "padded-patches"],
            "NOTES": """
## The block is a sandwich

```
h   = x + post_attention_layernorm(self_attn(input_layernorm(x)))
out = h + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(h)))
```

Four RMSNorms per block, no per-layer branch and no scalar.
`attention_output` and `mlp_output` are the post-norms' outputs, so
`vision.layers[i].input + attention_output + mlp_output == layer_output` holds exactly in float32,
and an ablation or a steering vector on a sublayer goes on `attention_output` or `mlp_output`.

## Queries, keys and values are normed, and the scores are not scaled

The attention norms each head's queries and keys (`q_norm`, `k_norm`) and its values (`v_norm`,
without gain), rotates queries and keys by the patch's x and y position, and computes the scores
with `scaling` 1.0. `model.vision.head_dim` is the config's `head_dim`: 64 on E2B and E4B, with 12
heads over a 768-wide stream.

## The padded rows run through every block

The processor resizes an image within a budget of `max_soft_tokens * pooling_kernel_size**2`
patches of 16 pixels and pads the row to that length: a square image is 48 × 48 = 2304 patches
and 216 padded rows of 2520. The padded rows are masked as keys, so no patch reads them, but they
are rows of every block's `layer_output` and of `tower_output`, with values. Select the image's
patches with the processor's positions:

```python
encoding = model.processor(text=prompt, images=[image], return_tensors="pt")
padded = (encoding["image_position_ids"] == -1).all(-1)   # [images, 2520]

with model.trace(dict(encoding)):
    stream = model.vision.layers[0].layer_output.save()

stream[~padded]                         # the 2304 patches
```

## The pooler makes the soft tokens

After the last block the vision encoder's `pooler` zeroes the padded rows, averages each 3 × 3
block of patches by position (`pooling_kernel_size`), multiplies by √`vision_hidden_size` and drops
the padding: 2304 patches become 256 soft tokens, `[soft_tokens, vision_hidden_size]` flat over the
images. These soft tokens are `model.projector.input`. A write to `tower_output` at the padded rows changes
nothing downstream; at the patches it reaches the text model. On 26B-A4B and 31B
(`standardize`) the pooled tokens are then shifted by `std_bias` and scaled by `std_scale`, inside
the vision encoder.
""",
        },
        "notes": """
## The projector reads the pooler's soft tokens

`embed_vision` reads the soft tokens the vision encoder's pooler averages from `tower_output`, not
`tower_output` itself.

## On 26B-A4B and 31B an image's tokens attend to each other on the sliding blocks

On a sliding block an image's tokens attend to every token of the same image within the window,
later ones included; the full blocks stay causal. E2B and E4B are causal throughout.
""",
    },
}

#: Set by hues.py (lineage: Gemma).
PALETTE = {"hue": 162}
QUIRKS = [
    "sandwich-norms", "per-block-sizes", "scaled-residual-adds", "qk-norm", "sliding-window", "proportional-rotary",
    # E2B and E4B (and the pinned tiny checkpoint) share keys and values; 26B-A4B and 31B set num_kv_shared_layers 0
    {"slug": "borrowed-kv", "when": lambda config: (config.num_kv_shared_layers or 0) > 0},
    # 26B-A4B (and the pinned tiny checkpoint) only
    {"slug": "mixture-of-experts", "when": lambda config: bool(config.enable_moe_block)},
    "scaled-embeddings",
    # 30 on every released size; null on the pinned tiny checkpoint
    {"slug": "softcapped-logits", "when": lambda config: config.final_logit_softcapping is not None},
]

#: What the visualization draws: the sublayers in forward order, each with its norms,
#: the standard value that is its contribution, and the values read inside it.
#: ``mlp`` is listed twice: a dense MLP on E2B, E4B and 31B, the dense MLP with a mixture of
#: experts beside it on 26B-A4B (and on the pinned tiny checkpoint).
BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "post_norm": "post_attention_layernorm",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} query heads over {num_kv_heads} key/value head(s); q, k and v normed per head, "
                      "scores not scaled",
            "variants": {
                "sliding_attention": "{sliding_window}-token window, head_dim {head_dim}",
                "full_attention": "full causal, wide heads, ¼ rotary",
            },
            "post_norm_note": "Follows the attention here; on Llama the same name is the MLP's pre-norm.",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "pre_feedforward_layernorm",
            "post_norm": "post_feedforward_layernorm",
            "contribution": "mlp_output",
            "detail": "GeGLU: {hidden_size} → {intermediate_size} → {hidden_size}, twice as wide on E2B's borrowing blocks",
        },
        {
            "host": "mlp",
            "kind": "moe",
            "label": "MLP + experts",
            "pre_norm": "pre_feedforward_layernorm",
            "post_norm": "post_feedforward_layernorm",
            "contribution": "mlp_output",
            "interior": ["router_logits", "expert_weights", "expert_indices", "expert_outputs", "routed_output"],
            "detail": "dense MLP + {num_experts} experts, top {top_k}",
            "pre_norm_note": "Feeds the dense MLP only; the router and the experts read the same stream through norms of their own.",
            "post_norm_note": "Norms post_feedforward_layernorm_1(mlp.output) + post_feedforward_layernorm_2(routed_output); "
                              "mlp.output is shared_expert_output.",
        },
    ],
    "identity": "(layers[i].input + self_attn.attention_output + mlp.mlp_output + per_layer_output) * layer_scalar == layer_output",
    "identity_note": "layer_scalar is the per-block buffer layers[i]._module.layer_scalar; per_layer_output exists on E2B and E4B only.",
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "Scaled by √hidden_size inside the embedding module, so token_embeddings equals layers[0].input on text. "
             "E2B and E4B also look each token up in embed_tokens_per_layer, which feeds per_layer_output.",
    "layers": "Each block multiplies its whole sum, the incoming stream included, by its layer_scalar.",
    "logits": "logits = 30 · tanh(lm_head.output / 30), which project_on_vocab applies too.",
}

NOTES = """
## Each block scales its whole sum by `layer_scalar`

```
h   = x + post_attention_layernorm(self_attn(input_layernorm(x)))
h   = h + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(h)))
h   = h + per_layer_output                # E2B and E4B
out = h * layer_scalar
```

`attention_output`, `mlp_output` and `per_layer_output` are the post-norms' outputs, before the
scalar, and on E2B every scalar is below one. A term's weight in the last stream is the product of
its own block's scalar and every later one, so direct logit attribution multiplies each term by that
product first, and a vector added to `mlp_output` reaches `layer_output` scaled by the block's scalar.

## Patching `token_embeddings` does not swap the token on E2B and E4B

Each block's per-layer input is a table lookup by token id plus a projection of `token_embeddings`,
so writing another token's embedding into `token_embeddings` changes only the projected half. To swap
a token, also patch `model.model.language_model.embed_tokens_per_layer.output` at that position.

## Borrowed keys and values are copies

On E2B and E4B the last blocks attend with an earlier block's keys and values, served to each
borrowing block as its own copy. An edit to the source block's `attention_keys` or `attention_values`
reaches only that block; an edit to its `k_proj` output reaches every block that borrows from it.

## On 26B-A4B the dense MLP runs beside the experts

`mlp.output` is the dense MLP's, served as `shared_expert_output`, and `mlp_output` norms it together
with `routed_output`. `expert_weights` are the top-8 softmax entries renormalized to sum to one, then
multiplied by the learned `per_expert_scale`.
"""
