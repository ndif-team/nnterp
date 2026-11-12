"""Gemma 4 unified, text: Gemma 4's sandwich block and learned block scale, with no per-layer embeddings and no
mixture of experts, behind an encoder-free image embedder."""

import os
import tempfile

MODEL_TYPE = "gemma4_unified_text"
TITLE = "Gemma 4 unified"

#: The public checkpoint the page opens on (meta build: config, tokenizer and processor, no weights).
REFERENCE = "google/gemma-4-12B"
#: The tiny checkpoint the suite's text tests run on.
PINNED = "hf-tiny-v2/tiny-random-Gemma4UnifiedForCausalLM"
#: Every checkpoint is a vision-language wrapper (model_type gemma4_unified, a key of WRAPPERS).
CHECKPOINTS = ["google/gemma-4-12B", "google/gemma-4-12B-it"]


def _tiny_wrapper() -> str:
    """The tiny wrapper ``tests/families/test_gemma4_unified_text.py`` builds (``_unified_checkpoint``): no tiny
    ``gemma4_unified`` checkpoint is published, so the suite writes one into the temp dir, named after the snapshot of
    the tiny text checkpoint it wraps. The same path, so the wrapper's ``pinned`` is the suite's ``REPO``."""
    from huggingface_hub import try_to_load_from_cache

    config = try_to_load_from_cache(PINNED, "config.json")  # the configured cache's snapshot, as the suite resolves it
    snapshot = os.path.basename(os.path.dirname(config)) if isinstance(config, str) else "missing"
    return os.path.join(tempfile.gettempdir(), f"nnterp-gemma4-unified-{snapshot}")


#: The vision-language wrapper of this family, keyed by the wrapper's config.model_type. Its image embedder is hosted
#: by this family alone, so it is inline (``tower``) rather than a module under encyclopedia/vision/.
WRAPPERS = {
    "gemma4_unified": {
        "title": "Gemma 4 unified",
        "pinned": _tiny_wrapper(),
        "projector": "embed_vision.multimodal_embedder: an RMS norm without gain (embedding_pre_projection_norm) and a "
                     "linear (embedding_projection) onto the text width, over every padded row of the embedder's states",
        "projector_input": "`vision.tower_output`: the embedder's states, the padded rows included",
        "tower": {
            "TITLE": "Gemma 4 unified embedder",
            "VISION_CONFIG_TYPES": ["gemma4_unified_vision"],
            "MODULE_CLASSES": ["Gemma4UnifiedVisionEmbedder"],
            # The embedder has no blocks (vision.num_layers == 0): there is no block to draw.
            "BLOCK": None,
            "ROWS": ("One row per image: its merged patches, then padding up to `max_soft_tokens` rows (280); the "
                     "processor's `image_position_ids` is `(-1, -1)` on the padded rows."),
            "MASKING": "None: the embedder has no attention, so no row reads another.",
            "POSITIONS": ("`patch_embeddings` is `patch_dense`'s output, before `patch_ln2`; a factorized 2D position "
                          "embedding (a learned x and a learned y row per patch, zero on the padded rows) is added after "
                          "`patch_ln2`, and `pos_norm` norms the sum."),
            "NORM": ("`pos_norm`, a LayerNorm, ends the embedder's states, so `vision.tower_output` is its output; "
                     "it is not `vision.norm`."),
            "QUIRKS": ["encoder-free", "variable-resolution", "padded-patches"],
            "NOTES": """
## The embedder is five modules and no blocks

```
h = patch_dense(patch_ln1(pixels))       # patch_embeddings
h = pos_norm(patch_ln2(h) + pos_x + pos_y)
out = multimodal_embedder(h)             # projector: RMS norm, linear
```

`model.vision` is `model.model.embed_vision`, a `Vision` with no `layers`: `vision.num_layers` is
0 and `vision.num_heads`, `head_dim` and `intermediate_size` raise `Unavailable`. Each row of
`pixel_values` is one merged patch of 48 × 48 pixels, 6912 values, and `patch_dense` maps it
straight to `vision.vision_hidden_size` (`mm_embed_dim`, 3840 on 12B). `patch_embeddings` and
`tower_output` are the embedder's only values.

## The padded rows run through the embedder

The processor fits each image into at most `max_soft_tokens` merged patches (280) and pads the
row to that length. The padded rows are rows of `patch_embeddings`, `tower_output` and
`projector.output`; the wrapper drops them before the scatter. Select the image's rows with the
processor's positions:

```python
encoding = model.processor(text=prompt, images=[image], return_tensors="pt")
valid = (encoding["image_position_ids"] != -1).all(-1)    # [images, 280]

with model.trace(dict(encoding)):
    states = model.vision.tower_output.save()

states[valid]                           # the image's merged patches
```
""",
        },
        "notes": """
## The projector reads `vision.tower_output`, padded rows included

The wrapper scatters only the image's rows of the projector's output, so a write to `tower_output`
at an image's rows reaches the text model and at the padded rows reaches nothing.

## An image's tokens attend to each other in both directions

An image's tokens attend to every token of the same image, later ones included, on the full blocks
and within the window on the sliding ones.
""",
    },
}

#: Set by hues.py (lineage: Gemma).
PALETTE = {"hue": 165}
QUIRKS = [
    "sandwich-norms", "per-block-sizes", "scaled-residual-adds", "qk-norm", "sliding-window", "proportional-rotary",
    "scaled-embeddings",
    # 30 on 12B; null on the pinned tiny checkpoint
    {"slug": "softcapped-logits", "when": lambda config: config.final_logit_softcapping is not None},
]

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
            "post_norm": "post_attention_layernorm",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} query heads over {num_kv_heads} key/value heads; q, k and v normed per head, "
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
            "detail": "GeGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_activation}",
        },
    ],
    "identity": "(layers[i].input + self_attn.attention_output + mlp.mlp_output) * layer_scalar == layer_output",
    "identity_note": "layer_scalar is the per-block buffer layers[i]._module.layer_scalar.",
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "Scaled by √hidden_size inside the embedding module, so token_embeddings equals layers[0].input on text.",
    "layers": "Each block multiplies its whole sum, the incoming stream included, by its layer_scalar.",
    "logits": "logits = 30 · tanh(lm_head.output / 30), which project_on_vocab applies too.",
}

NOTES = """
## Each block scales its whole sum by `layer_scalar`

```
h   = x + post_attention_layernorm(self_attn(input_layernorm(x)))
h   = h + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(h)))
out = h * layer_scalar
```

`attention_output` and `mlp_output` are the post-norms' outputs, before the scalar, and the scalars
are far from one on 12B. A term's weight in the last stream is the product of its own block's scalar
and every later one, so direct logit attribution multiplies each term by that product first, and a
vector added to `mlp_output` reaches `layer_output` scaled by the block's scalar.
"""
