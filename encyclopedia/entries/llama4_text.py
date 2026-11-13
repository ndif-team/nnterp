"""Llama 4, text: Llama's block with dense and mixture blocks interleaved, NoPE blocks every fourth, and chunked attention on the rest."""

MODEL_TYPE = "llama4_text"
TITLE = "Llama 4"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "meta-llama/Llama-4-Scout-17B-16E"
#: The tiny checkpoint the test suite builds the page from. The suite rewrites its config into a local copy
#: (``attn_temperature_tuning`` from the int 4 to ``true``, which transformers' config validation wants as a bool), and the
#: page is built from that copy. It is a ``llama4`` wrapper with a four-block ``llama4_text`` model.
PINNED = "yujiepan/llama-4-tiny-random"
#: Every published checkpoint is a ``llama4`` wrapper (Llama4ForConditionalGeneration), a key of WRAPPERS, so the
#: page opens on a vision-language checkpoint; the text-generation task builds Llama4ForCausalLM from the same repo.
CHECKPOINTS = [
    "meta-llama/Llama-4-Scout-17B-16E", "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "meta-llama/Llama-4-Maverick-17B-128E", "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
]

#: The vision-language wrapper of this family, keyed by the wrapper's config.model_type. Its vision encoder is
#: hosted by this family alone, so it is inline (``tower``) rather than a module under encyclopedia/vision/.
#: Every shape and identity below was checked on the pinned tiny wrapper (config-patched, bfloat16, on CPU);
#: the sizes are Scout's and Maverick's configs (their vision configs are the same).
WRAPPERS = {
    "llama4": {
        "title": "Llama 4",
        "pinned": "yujiepan/llama-4-tiny-random",
        "projector": "a linear (linear_1, no bias) over the pixel-shuffle adapter's output, flattened over the tiles",
        "projector_input": "the pixel-shuffle adapter's output over `vision.tower_output`, flattened over the tiles",
        "quirks": ["tiled-images", "pooled-projector"],
        "tower": {
            "TITLE": "Llama 4 ViT",
            "VISION_CONFIG_TYPES": ["llama4_vision_model"],
            "MODULE_CLASSES": ["Llama4VisionModel"],
            "BLOCK": {
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
                        "detail": "{num_heads} heads × {head_dim}, 2D rotary",
                    },
                    {
                        "host": "mlp",
                        "kind": "mlp",
                        "label": "MLP",
                        "pre_norm": "post_attention_layernorm",
                        "contribution": "mlp_output",
                        "detail": "GELU: {hidden_size} → {intermediate_size} → {hidden_size}",
                    },
                ],
            },
            "ROWS": ("One row per image tile: the processor fits an image onto a grid of up to `max_patches` (16) "
                     "tiles of 336 pixels and adds a global tile, the whole image resized, when there is more than "
                     "one. A row is the tile's 576 patches, then the CLS token last: `patches + 1` rows."),
            "MASKING": "No mask: each row attends to every patch of its own tile and to its CLS token; tiles do not see one another.",
            "POSITIONS": ("`patch_embeddings` is the unfold-and-linear output, before the CLS token is appended. A learned "
                          "embedding per row (`positional_embedding_vlm`, the CLS's last) is added and `layernorm_pre` "
                          "norms the rows; every attention rotates queries and keys by the patch's 2D position, and "
                          "leaves the CLS row unrotated."),
            "NORM": ("`layernorm_post` is `vision.norm`, a LayerNorm over every row, CLS included; `vision.tower_output` "
                     "is its output. The tower then drops the CLS row and runs `vision.vision_adapter`."),
            "QUIRKS": ["cls-token"],
            "NOTES": """
## The block is LayerNorm pre-norm, with biases

```
h   = x + self_attn(input_layernorm(x))             # 2D rotary on q and k
out = h + mlp(post_attention_layernorm(h))          # fc1, GELU, fc2
```

Both norms are LayerNorms, and the projections and both MLP linears carry biases. Nothing norms a
sublayer's output, so `vision.layers[i].input + attention_output + mlp_output == layer_output`.
On Scout and Maverick the vision encoder is 34 blocks of 1408 wide, 16 heads of 88.

## The CLS token is the last row

The tower appends its `class_embedding` after the 576 patches, so every `vision.layers[i]` value
is `[tiles, 577, vision_hidden_size]` with the CLS at index `-1`, and `patch_embeddings` is the 576
patches alone. `layernorm_post` norms all 577 rows; the tower then drops the CLS, so
`vision.tower_output[:, -1]` reaches nothing: on the pinned tiny checkpoint a write to it leaves
the logits bit-identical, and a write to `vision.tower_output[:, :-1]` changes them. Select the
patches with `[:, :-1]`, not CLIP's `[:, 1:]`.

## The pixel-shuffle adapter is inside the vision encoder

`vision.vision_adapter` takes `vision.tower_output[:, :-1]`, merges each 2 × 2 block of patches
into one row (`pixel_shuffle_ratio` 0.5) and runs a two-layer MLP with a GELU after each linear:
576 patches become 144 rows of `projector_output_dim` (4096). Its output is the tower's own
`last_hidden_state`.

```python
with model.trace(prompt, images=[image]):
    out = model.vision.tower_output.save()              # [tiles, 577, vision_hidden_size]
    read = model.vision.vision_adapter.input.save()
    shuffled = model.vision.vision_adapter.output.save()

torch.equal(read, out[:, :-1])     # True: the patches, the CLS dropped
shuffled.shape[1]                  # 144 on Scout: 576 / 4
```
""",
        },
        "notes": """
## The projector reads 144 rows per tile

`projector.input` is `vision.vision_adapter`'s output flattened over the tiles, so an image's count of
image tokens depends on how many tiles the processor cuts it into.
""",
    },
}

#: Set by hues.py (lineage: Llama and its kin).
PALETTE = {"hue": 244}
QUIRKS = [
    # Scout (use_qk_norm true); Maverick's rotary blocks have no qk_norm
    {"slug": "qk-norm", "when": lambda config: config.use_qk_norm},
    "nope-blocks", "chunked-attention", "mixture-of-experts",
    # Maverick (interleave_moe_layer_step 2); Scout's step is 1, a mixture on every block
    {"slug": "interleaved-moe", "when": lambda config: config.interleave_moe_layer_step > 1},
]

#: What the visualization draws: the sublayers in forward order, each with its norms,
#: the standard value that is its contribution, and the values read inside it.
#: ``mlp`` is listed twice: a dense MLP on Maverick's even blocks, a mixture with a shared expert on
#: Maverick's odd blocks and on every Scout block (the pinned tiny checkpoint: dense 0 and 2, mixture 1 and 3).
#: ``expert_weights`` and ``expert_outputs`` are unavailable on every Llama 4 mixture, so no chip draws them.
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
            "detail": "{num_heads} query heads over {num_kv_heads} key/value heads",
            "variants": {
                "chunked_attention": "{num_heads}/{num_kv_heads} heads, rotary, chunked",
                "full_attention": "{num_heads}/{num_kv_heads} heads, NoPE, full causal",
            },
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size_mlp} → {hidden_size}",
            "pre_norm_note": "The MLP's input norm, named for its place after the attention's add.",
        },
        {
            "host": "mlp",
            "kind": "moe",
            "label": "MoE",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "interior": ["router_logits", "expert_indices", "routed_output", "shared_expert_output"],
            "detail": "{num_experts} experts + shared, top {top_k}",
            "pre_norm_note": "The mixture's input norm, named for its place after the attention's add.",
        },
    ],
}

#: Notes on the model-level strip, by node: none, since nothing happens at the embeddings, the final norm or the head.
STRIP: dict[str, str] = {}

NOTES = """
## The router scales the expert's input, not its output

```
s   = sigmoid(top_1(router_logits))       # 0 for every expert but the chosen one
out = shared_expert(x) + experts(x * s)
```

Each token's one expert reads the token's input times its sigmoid score, so `expert_weights` and
`expert_outputs` are unavailable, and `mlp_output` is `shared_expert_output + routed_output`. Steer the
routing at `router_logits`, where an expert's logit at `-inf` sends its tokens to their next choice.
"""
