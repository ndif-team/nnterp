"""Llama 4, text: Llama's block with dense and mixture blocks interleaved, NoPE blocks every fourth, and chunked attention on the rest."""

MODEL_TYPE = "llama4_text"
TITLE = "Llama 4"
SUBTITLE = (
    "Llama's pre-norm block where the MLP is a mixture of experts with a shared expert (on every block of Scout, "
    "every other block of Maverick) routed by a sigmoid over the top-1 logit, every fourth block attends without "
    "rotary over the whole context with temperature-tuned queries, and the rest attend within chunks."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "meta-llama/Llama-4-Scout-17B-16E"
#: The tiny checkpoint the test suite builds the page from. The suite rewrites its config into a local copy
#: (``attn_temperature_tuning`` from the int 4 to ``true``, which transformers 5.17 validates as a bool), and the
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
is `[tiles, 577, vision_hidden]` with the CLS at index `-1`, and `patch_embeddings` is the 576
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
    out = model.vision.tower_output.save()              # [tiles, 577, vision_hidden]
    read = model.vision.vision_adapter.input.save()
    shuffled = model.vision.vision_adapter.output.save()

torch.equal(read, out[:, :-1])     # True: the patches, the CLS dropped
shuffled.shape[1]                  # 144 on Scout: 576 / 4
```
""",
        },
        "notes": """
## The projector reads the adapter's output, flattened over the tiles

`model.multi_modal_projector` (the `projector`) is one linear, 4096 to the text model's 5120,
without a bias. Its input is `vision.vision_adapter`'s output flattened over the tiles,
`[tiles × 144, 4096]`, so `projector.input` has a quarter as many rows per tile as the tile has
patches. `vision.image_features` is `model.projector.output`, and
`layers[0].input[vision.image_token_mask] == vision.image_features` holds exactly. On the pinned
tiny checkpoint:

```python
with model.trace(prompt, images=[image]):
    mask = model.vision.image_token_mask.save()
    shuffled = model.vision.vision_adapter.output.save()
    fed = model.projector.input.save()
    features = model.vision.image_features.save()
    first = model.layers[0].input.save()

torch.equal(fed, shuffled.flatten(0, 1))     # True: the adapter's rows, tile after tile
torch.equal(first[mask], features)           # True: the scatter
```

A write to `vision.tower_output` at the patch rows reaches the adapter and the text model.

## 144 image tokens per tile

`config.image_token_index` is `<|patch|>`: the processor writes 144 of them per tile, with
`<|tile_x_separator|>` and `<|tile_y_separator|>` between tiles and the global tile's 144 last,
and `vision.image_token_mask` marks the `<|patch|>` positions only. A 64 × 64 image is one tile
and 144 image tokens; a 900 × 500 image is a 2 × 3 grid plus the global tile, 7 rows of the
vision encoder's batch and 1008 image tokens. The count depends on the image's shape: assign one
image's features into another's run only when the two give the same count.

## The wrapper's own forward scatters

`Llama4ForConditionalGeneration` has no inner model: its own `forward` runs the vision encoder,
the projector and `inputs_embeds.masked_scatter`, and `vision.image_features` is read at that
scatter on the root. The text model is `model.language_model`, with `lm_head` under it.

## Load in bfloat16

The image processor returns bfloat16 pixels, and a float32 vision encoder refuses them
(`expected m1 and m2 to have the same dtype`). Load with `dtype=torch.bfloat16`, or build the
encoding with `model.processor` and cast `pixel_values` to the model's dtype before the trace.
""",
    },
}

#: Set by hues.py (lineage: Llama and its kin).
PALETTE = {"hue": 244}
VLLM = False
QUIRKS = ["qk-norm", "nope-blocks", "chunked-attention", "mixture-of-experts", "interleaved-moe"]

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
            "detail": "SwiGLU, intermediate_size_mlp wide",
            "pre_norm_note": "The MLP's input norm, as on Llama.",
        },
        {
            "host": "mlp",
            "kind": "moe",
            "label": "MoE",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "interior": ["router_logits", "expert_indices", "routed_output", "shared_expert_output"],
            "detail": "{num_experts} experts + shared, top {top_k}",
            "pre_norm_note": "The mixture's input norm, as on Llama: the router, every expert and the shared expert read its output.",
        },
    ],
    "identity_note": "Exact on every block of the pinned tiny checkpoint, dense and mixture. mlp_output is the block's "
                     ".view(residual.shape) of the feed-forward's output, which a mixture returns flattened to [batch * seq, hidden].",
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "A plain lookup with no scale: on a prompt without an image, token_embeddings equals layers[0].input.",
    "layers": "Every fourth block (3, 7, …, 47) is a NoPE block attending over the whole context; the others rotate "
              "queries and keys and attend within chunks of 8192 tokens. Scout has a mixture of experts on every block, "
              "Maverick on the odd ones.",
    "norm": "Llama4TextRMSNorm, the gain norm.weight itself, eps 1e-5; project_on_vocab applies it.",
    "head": "lm_head has its own weight on Scout and Maverick (tie_word_embeddings false); logits is lm_head.output.",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))               # rotary + L2 qk-norm, or NoPE
f   = feed_forward(post_attention_layernorm(h))       # dense, or (out, router_logits)
out = h + f.view(h.shape)
```

Llama's pre-norm block with the feed-forward named `feed_forward`, which nnterp calls `mlp`;
`post_attention_layernorm` is the MLP's input norm. Nothing norms a sublayer's output, and the
identity is the plain sum on dense and mixture blocks alike, exact on the pinned tiny checkpoint:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.equal(x + attn + mlp, out)    # True
```

## `mlp_output` is the block's view, not the module's output

A mixture returns `(out, router_logits)` with `out` flattened to `[batch * seq, hidden]`, and the
block views it back into the residual's shape before the add. `mlp_output` is read at that view,
`[batch, seq, hidden]` on every block, and an in-place edit of it lands on the mixture's own output
tensor; `layers[i].mlp.output[0]` is the flat tensor. The view is an operation of the block's
forward, read after the attention has returned, so the family's `Layer` is `sourced`: nnterp
instruments the block's forward when the envoy is built.

## Dense and mixture blocks interleave

`interleave_moe_layer_step` places the mixtures (`config.moe_layers`): 1 on Scout, so all 48
blocks are a `Moe`, and 2 on Maverick, so blocks 1, 3, …, 47 are a `Moe` and the even blocks a dense
`Mlp`. The widths are spelled the other way round from most families: the config's
`intermediate_size` (8192) is each expert's and the shared expert's, and `intermediate_size_mlp`
(16384) is the dense MLP's, which `model.intermediate_size` reports. On Scout no block uses that
width. On a dense block every mixture value is missing, with the reason saying so.

## The router takes a sigmoid over the top-1 logit

`num_experts_per_tok` is 1 on both: Scout routes over 16 experts, Maverick over 128, and every
mixture also runs its shared expert on every token. The router keeps the largest logit, sets the
others to `-inf` and takes a sigmoid (`SCORING` `"sigmoid"`), so a token's score is between 0 and 1
for its expert and 0 for the rest. Each expert's *input* is multiplied by its score, every expert
runs on every token, and the routed sum is added in place to the shared expert's output. So
`router_logits` (the router's projection, `[batch, seq, experts]`), `expert_indices` (its top 1),
`routed_output` (the sum over experts) and `shared_expert_output` (a copy, whose edits are carried
back) are served, and `expert_weights` and `expert_outputs` are unavailable.

```python
moe = model.layers[1].mlp
with model.trace(prompt):
    logits = moe.router_logits.save()
    idx = moe.expert_indices.save()             # [batch, seq, 1]
    shared = moe.shared_expert_output.save()
    routed = moe.routed_output.save()
    contribution = moe.mlp_output.save()

torch.equal(logits.topk(moe.top_k, dim=-1).indices, idx)   # True
torch.testing.assert_close(shared + routed, contribution)
```

## Steer the routing at `router_logits`

`expert_indices` and the scores are both computed from `router_logits`, so a write there moves
both. Setting expert `e`'s logit to `-inf` sends the tokens that chose it to their second choice,
and `mlp_output` changes on exactly those tokens; setting it high sends every token to `e`.
`moe.routed_output[:] = 0` leaves the shared expert's output as the block's whole `mlp_output`, and
`moe.shared_expert_output[:] = 0` leaves the routed sum.

```python
with model.trace(prompt):
    moe.router_logits[..., e] = float("-inf")   # e's tokens go to their next expert
    rerouted = moe.mlp_output.save()
```

## Every fourth block has no rotary and tunes its queries

`no_rope_layers` (built from `no_rope_layer_interval` 4) makes blocks 3, 7, …, 47 NoPE blocks:
no rotary embedding and no query/key norm, and full causal attention (`layer_types`
`full_attention`). On them `attn_temperature_tuning` multiplies the queries by
`1 + attn_scale · log(1 + floor((position + 1) / floor_scale))`, with `attn_scale` 0.1 and
`floor_scale` 8192 on both, so the factor is exactly 1 for the first 8191 positions and steps up
by 0.069 at 8191. The other blocks (`chunked_attention`) rotate queries and keys and attend only
within their chunk of `attention_chunk_size` 8192 tokens, so their pattern is zero across chunk
boundaries. `attention_queries` and `attention_keys` are read after the rotary, the norm and the
temperature: on a NoPE block of a prompt shorter than 8191 tokens, `attention_queries` is
`q_proj`'s output split into heads.

## The query/key norm is an L2 norm, on Scout only

Where `use_qk_norm` is true, a rotary block divides each query and key head by its RMS after the
rotary (`qk_norm`, a `Llama4TextL2Norm` with no weight), so every head of `attention_queries` and
`attention_keys` has RMS 1. One module serves both, and NoPE blocks have none. Scout sets
`use_qk_norm`; Maverick does not, and its blocks have no `qk_norm`. The scores are scaled by
`1/√head_dim` on every block. Both have 40 query heads over 8 key/value heads of 128, so an edit
to one head of `attention_keys` or `attention_values` reaches 5 query heads.

## Rotary and context

Scout rotates with `rope_theta` 500,000 under llama3 scaling (`factor` 16 over an original 8192
positions; `max_position_embeddings` 10,485,760); Maverick with the same base, unscaled
(`max_position_embeddings` 1,048,576). NoPE blocks carry no position: only the causal mask orders
the tokens there.

## The readout

`logits` is `lm_head.output`, with no cap or scale, and `lm_head` is its own weight on both
checkpoints (`tie_word_embeddings` false). The vocabulary is 202,048 tokens. The final norm is an
RMSNorm whose gain is `norm.weight` itself; `project_on_vocab` applies it and `lm_head`.

## Every checkpoint is a multimodal wrapper

The published repos are `llama4` checkpoints with a `llama4_text` `text_config`. The
text-generation task builds `Llama4ForCausalLM` from the text config, reading its weights out of the
wrapper checkpoint and leaving the vision encoder's unused; `task="image-text-to-text"` builds
`Llama4ForConditionalGeneration` with its processor, the text model under `language_model`. The
standard names are the same under both loads.
"""
