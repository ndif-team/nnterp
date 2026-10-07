"""Gemma 4, text: a sandwich block with per-layer embeddings, a learned scale on each block's sum, and borrowed keys and values."""

MODEL_TYPE = "gemma4_text"
TITLE = "Gemma 4"
SUBTITLE = (
    "A sandwich block whose sum is multiplied by a learned per-block scalar, with a third add from per-layer "
    "embeddings, normed queries, keys and values, and keys and values the last blocks borrow from earlier ones."
)

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
                        "post_norm_note": "This norm follows the attention, as on the text block: its output is attention_output.",
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
block of patches by position (`pooling_kernel_size`), multiplies by √`vision_hidden` and drops
the padding: 2304 patches become 256 soft tokens, `[soft_tokens, vision_hidden]` flat over the
images. That is `model.projector.input`. A write to `tower_output` at the padded rows changes
nothing downstream; at the patches it reaches the text model. On 26B-A4B and 31B
(`standardize`) the pooled tokens are then shifted by `std_bias` and scaled by `std_scale`, inside
the vision encoder.
""",
        },
        "notes": """
## `embed_vision` projects the pooled soft tokens

`model.embed_vision` (the `projector`) receives the pooler's soft tokens, not `tower_output`:
`model.projector.input` is `[soft_tokens, vision_hidden]`, flat over the images. It norms them
without a gain (`embedding_pre_projection_norm`) and projects them to the text model's width
(`embedding_projection`); `vision.image_features` is `model.projector.output`, and
`layers[0].input[vision.image_token_mask] == vision.image_features` holds exactly. On the pinned
tiny checkpoint:

```python
with model.trace(prompt, images=[image]):
    mask = model.vision.image_token_mask.save()
    out = model.vision.tower_output.save()
    fed = model.projector.input.save()
    features = model.vision.image_features.save()
    first = model.layers[0].input.save()

out.shape[1], fed.shape[0]               # 2520 rows, 256 soft tokens
torch.equal(first[mask], features)       # True: the scatter
```

## 256 image tokens for a square image

On `google/gemma-4-E2B` a 224 × 224 image is 2304 patches and 216 padded rows: under eager every
vision encoder block's pattern is `(1, 12, 2520, 2520)`, and the text model receives 256 image tokens,
`vision.image_features` `(256, 1536)`. The trace ran under `torch.no_grad()` in bfloat16 and peaked
at 11 GB. An image of another shape gets another grid in the same 2520 rows (a 96 × 48 image is 2277
patches and 253 image tokens), so the count of image tokens depends on the image: assign one
image's features into another's run only when the two give the same count.

## The base checkpoints have no chat template

`google/gemma-4-E2B` ships no chat template: put the processor's image token in the prompt
yourself, `f"{model.processor.image_token} What is this?"`.

## The audio side keeps its native names

E2B and E4B also carry an audio encoder. `model.model.audio_tower` and `model.model.embed_audio`
keep their native names, and an audio prompt, `model.trace(prompt, audio=[waveform])`, runs under
the same load.
""",
    },
}

#: Set by hues.py (lineage: Gemma).
PALETTE = {"hue": 162}
VLLM = False
QUIRKS = [
    "sandwich-norms", "scaled-residual-adds", "qk-norm", "sliding-window", "proportional-rotary", "borrowed-kv",
    "mixture-of-experts", "scaled-embeddings", "softcapped-logits",
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
                      "scores not scaled; the last {num_kv_shared_layers} blocks borrow keys and values",
            "variants": {
                "sliding_attention": "{sliding_window} window, head_dim {head_dim}",
                "full_attention": "full causal, wide heads, ¼ rotary",
            },
            "post_norm_note": "As on Gemma 2 and 3, this norm follows the attention. On Llama the same name is the norm before the MLP.",
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
            "pre_norm_note": "On a mixture block this norm feeds the dense MLP only: the router reads the stream before it, "
                             "and the experts read pre_feedforward_layernorm_2's output of that stream.",
            "post_norm_note": "On a mixture block this norm norms post_feedforward_layernorm_1(mlp.output) + "
                              "post_feedforward_layernorm_2(routed_output); mlp.output is shared_expert_output.",
        },
    ],
    "identity": "(layers[i].input + self_attn.attention_output + mlp.mlp_output + per_layer_output) * layer_scalar == layer_output",
    "identity_note": "layer_scalar is layers[i]._module.layer_scalar, a per-block buffer; per_layer_output is the block's own value. "
                     "Exact in float32 on every E2B block. 26B-A4B and 31B have no per-layer embeddings, and the term drops out.",
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "embed_tokens multiplies its lookup by √hidden_size, cast to the weight's dtype (39.19 on E2B, 39.25 in bfloat16), "
             "so token_embeddings is the scaled tensor and equals layers[0].input. E2B and E4B also look each token up in "
             "embed_tokens_per_layer, which feeds per_layer_output.",
    "layers": "Each block ends by multiplying its sum by layer_scalar (0.019 to 0.88 on E2B), so the stream shrinks "
              "block by block and a block's terms reach the last stream multiplied by every scalar from there on.",
    "norm": "Gemma4RMSNorm multiplies by norm.weight itself, not 1 + weight; the final norm's weights average 14.2 on E2B (maximum 118.5).",
    "head": "lm_head shares its weight with embed_tokens (tie_word_embeddings).",
    "logits": "final_logit_softcapping is 30 on every released size: logits is 30 · tanh(lm_head.output / 30), and project_on_vocab applies the cap.",
}

NOTES = """
## The block, in order

```
a   = input_layernorm(x)
q   = rope(q_norm(q_proj(a)))         # per head
k   = rope(k_norm(k_proj(a)))         # per head; borrowed on the last blocks
v   = v_norm(v_proj(a))               # per head, no gain
h   = x + post_attention_layernorm(o_proj(attend(q, k, v)))      # no 1/√d
h   = h + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(h)))
p   = per_layer_projection(gelu(per_layer_input_gate(h)) * per_layer_input)
h   = h + post_per_layer_input_norm(p)
out = h * layer_scalar
```

Four RMSNorms around the sublayers as on Gemma 2 and 3, a fifth after the per-layer branch, and
three inside the attention. `post_attention_layernorm` follows the attention; the MLP's input
norm is `pre_feedforward_layernorm`. The per-layer branch has no module of its own: its three
modules are children of the block, and `per_layer_output` is a value of `layers[i]`.
`layer_scalar` is a buffer, `layers[i]._module.layer_scalar`, applied in place.

## The block's sum is scaled; its terms are served unscaled

`attention_output`, `mlp_output` and `per_layer_output` are the post-norms' outputs, before
`layer_scalar`. The identity is exact in float32 on every E2B block:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    ple = model.layers[1].per_layer_output.save()
    out = model.layers[1].layer_output.save()

scalar = model.layers[1]._module.layer_scalar
torch.testing.assert_close((x + attn + mlp + ple) * scalar, out)
```

No scalar is 1 on E2B: they run from 0.019 (block 0) to 0.88 (block 30), with 0.074 and 0.049 at
blocks 13 and 14 and 0.16 at the last block. A vector added to `mlp_output` reaches `layer_output`
multiplied by the block's scalar; one added to `layer_output` does not.

## A term's weight in the last stream is the product of the scalars from its block on

Block `i`'s terms are multiplied by its own scalar and by every later one; the embeddings by all
of them. In float32 the last block's output is that weighted sum:

```python
scalars = torch.cat([layer._module.layer_scalar for layer in model.layers])
weight = scalars.flip(0).cumprod(0).flip(0)

added = []
with model.trace(prompt):
    emb = model.token_embeddings.save()
    for layer in model.layers:
        attn = layer.self_attn.attention_output
        added.append((attn + layer.mlp.mlp_output + layer.per_layer_output).save())
    final = model.layers[-1].layer_output.save()

total = weight[0] * emb + sum(w * a for w, a in zip(weight, added))
torch.testing.assert_close(total, final)
```

On E2B the weights are 1.9e-13 for block 0 (and the embeddings), 1.8e-6 for block 14, 1.2e-3 for
block 20, 0.06 for block 30 and 0.16 for block 34. The final norm divides out the common scale,
so only the ratios count: block 34's terms outweigh block 0's by about 10¹². At the last position
of a 9-token prompt, block 0's `attention_output` has norm 1476 and block 34's `mlp_output` 122;
weighted, they are 2.8e-10 and 20 in a last stream of norm 28. Direct logit attribution multiplies
each term by its weight first; the early blocks act on the logits only through the blocks that
read the stream after them.

## Per-layer embeddings add a third term

E2B and E4B (`hidden_size_per_layer_input` 256) give every block a 256-wide input of its own.
The model looks each token up in `model.model.language_model.embed_tokens_per_layer`
(`[262144, 35 × 256]` on E2B, scaled by 16: 2.35B of the language model's 4.63B parameters), adds
a normed projection of `token_embeddings`, and scales the sum by 2^-1/2; block `i` receives slice
`i` as its second argument, `model.layers[i].inputs[0][1]` (`[batch, seq, 256]`). The block gates
it by `gelu(per_layer_input_gate(h))`, projects it to the hidden size and norms it:
`per_layer_output`. The model depends on it: zeroing it on every block takes p(" Paris") after
"The Eiffel Tower is in the city of" from 0.91 to 6e-10 on E2B.

```python
with model.trace(prompt):
    for layer in model.layers:
        layer.per_layer_output[:] = 0
    logits = model.logits.save()
```

## Patching `token_embeddings` does not swap the token

The per-layer table is looked up by token id, not from `token_embeddings`, so writing another
token's embedding into `token_embeddings` changes only the projected half of every block's
per-layer input. On E2B, putting " Rome"'s embedding in place of " Paris"'s at the end of "The
capital of France is Paris" leaves a KL of 0.51 to the " Rome" run. Patching the table's output
as well reproduces that run exactly:

```python
with model.trace(rome):
    emb = model.token_embeddings.save()
    table = model.model.language_model.embed_tokens_per_layer.output.save()

with model.trace(paris):
    model.token_embeddings[:, -1] = emb[:, -1]
    model.model.language_model.embed_tokens_per_layer.output[:, -1] = table[:, -1]
    logits = model.logits.save()
```

## Queries, keys and values are normed, and the scores are not scaled

`q_norm` and `k_norm` norm each head before the rotary; `v_norm` norms each value head with no
gain, so every head of `attention_values` has RMS 1. The attention's `scaling` is 1.0: the scores
are `q · k` with no `1/√head_dim`, and the norms' gains set the temperature. `attention_queries`
and `attention_keys` are read after their norms and the rotary; `q_norm.output` is the normed
queries before it. E2B has one key/value head for its 8 query heads, so an edit to
`attention_keys` or `attention_values` reaches all 8; E4B has two, each serving 4.

## Sliding and full blocks differ in width and rotary

`config.layer_types` makes every fifth block full on E2B (4, 9, …, 34) and every sixth on E4B,
26B-A4B and 31B; the last block is full on all four. The window is 512 tokens on E2B and E4B (a
query attends to itself and the 511 before it) and 1024 on 26B-A4B and 31B. Full blocks have
512-wide heads against the sliding blocks' 256, on every size; 26B-A4B and 31B also give them
fewer key/value heads (2 against 8, 4 against 16). The root's sizes are the sliding blocks'; each
block's own are on its attention and MLP:

```python
model.head_dim, model.layers[4].self_attn.head_dim        # 256, 512 on E2B
model.layers[4].self_attn.num_heads                       # 8: the same on every block
model.layers[20].mlp.intermediate_size                    # 12288 on E2B; 6144 at the root
```

Sliding blocks rotate all 256 dimensions with base 10,000. Full blocks use proportional rotary:
only dimensions 0 to 63 and 256 to 319 of the 512 turn, at frequencies spaced over the whole head
with base 1,000,000 (the slowest has a wavelength of 188 tokens); the other 384 carry no position.

## Blocks 15 to 34 attend with borrowed keys and values on E2B

The last `num_kv_shared_layers` blocks (20 of 35 on E2B, 18 of 42 on E4B, none on 26B-A4B and
31B) have no `k_proj`, `v_proj`, `k_norm` or `v_norm`. On E2B the sliding ones attend with block
13's keys and values and the full ones with block 14's. Their `attention_keys` are block 13's or
14's, as a copy private to the block: zeroing block 13's leaves block 20's as they were, while an
edit of block 13's `k_proj` output reaches every sliding block that borrows from it.

```python
with model.trace(prompt):
    source = model.layers[13].self_attn.attention_keys.save()
    borrowed = model.layers[20].self_attn.attention_keys.save()

assert torch.equal(source, borrowed)
```

`skip_layers` over block 13 or 14 leaves the borrowers nothing to read and fails with a `KeyError`
inside transformers.

## Eager and sdpa compute the same attention

Gemma 4 has no attention softcap, and the scale is 1.0 under every implementation. On E2B in
float32 a default (`sdpa`) load's logits are within 2.1e-5 of an eager load's on a 9-token prompt
and 3.5e-5 on an 881-token one. `attn_implementation="eager"` is needed only for the values read
inside the attention function (the ones marked `⚠`).

## The readout is softcapped, and the norm gain is the weight

`logits` is `30 · tanh(lm_head.output / 30)` on every released size; `project_on_vocab` applies the
final norm, `lm_head` and the cap, so a lens at the last block reproduces `logits`. The
vocabulary is 262,144 tokens on every size, and `lm_head` is `embed_tokens`' weight. Unlike Gemma 2
and 3, `Gemma4RMSNorm` multiplies by `weight`, not `1 + weight`: folding the final norm into
`lm_head` uses `model.norm._module.weight` as it is (mean 14.2 on E2B, maximum 118.5).

## Every checkpoint is a multimodal wrapper

The released repos are `gemma4` checkpoints: the text-generation task builds
`Gemma4ForConditionalGeneration` (vision and audio towers on E2B and E4B, vision only on 26B-A4B
and 31B), nnterp picks this family from `config.text_config`, and the text stack sits under
`model.model.language_model` with `lm_head` at the root. `Gemma4ForCausalLM` is built only from a
`gemma4_text` config, as the suite's tiny checkpoint is. `google/gemma-4-12B` is a
`gemma4_unified` checkpoint and loads as `gemma4_unified_text`: the same block without per-layer
embeddings or a mixture of experts.

## 26B-A4B runs a mixture of experts beside the dense MLP

Under `enable_moe_block` every block runs 128 experts, 8 per token and each 704 wide, beside the
2112-wide dense MLP. The router reads the stream after the attention add, through its own norm
without gain, scales it by `router.scale` and `hidden_size ** -0.5`, takes a softmax over the 128
experts and keeps the top 8, renormalised to sum to one and then multiplied by `per_expert_scale`:
`expert_weights` are those products. The experts read `pre_feedforward_layernorm_2` of the same
stream. `shared_expert_output` is the dense MLP's output, `mlp.output`, and the two meet in
three norms:

```
mlp_output = post_feedforward_layernorm(
    post_feedforward_layernorm_1(shared_expert_output)
    + post_feedforward_layernorm_2(routed_output))
```

On E2B, E4B and 31B every mixture value is unavailable.
"""
