"""Gemma 4 unified, text: Gemma 4's sandwich block and learned block scale, with no per-layer embeddings and no
mixture of experts, behind an encoder-free image embedder."""

import glob
import os
import tempfile

MODEL_TYPE = "gemma4_unified_text"
TITLE = "Gemma 4 unified"
SUBTITLE = (
    "A sandwich block whose sum is multiplied by a learned per-block scalar, with normed queries, keys and values, "
    "and full blocks that take their values from the key projection."
)

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
    snapshots = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{PINNED.replace('/', '--')}/snapshots/*"))
    snapshot = os.path.basename(snapshots[0]) if snapshots else "missing"
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
straight to `vision.hidden_size` (`mm_embed_dim`, 3840 on 12B). `patch_embeddings` and
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
## `image_features` is the projector's output without the padded rows

`model.projector` is `embed_vision.multimodal_embedder`, and it receives `vision.tower_output`
itself: `projector.input` is `[images, 280, mm_embed_dim]`, padded rows included. The wrapper
keeps the rows where `image_position_ids` is not `(-1, -1)` and scatters them, so
`vision.image_features == projector.output[valid]`, and
`layers[0].input[vision.image_token_mask] == vision.image_features` holds exactly. On the pinned
tiny wrapper:

```python
valid = (encoding["image_position_ids"] != -1).all(-1)

with model.trace(dict(encoding)):
    mask = model.vision.image_token_mask.save()
    projected = model.projector.output.save()
    features = model.vision.image_features.save()
    first = model.layers[0].input.save()

torch.equal(projected[valid], features)   # True
torch.equal(first[mask], features)        # True: the scatter
```

A write to `tower_output` at an image's rows reaches the text model; at the padded rows it
reaches nothing.

## One image token per merged patch

An image is as many image tokens as it has merged patches, at most 280. With
`google/gemma-4-12B`'s processor a square image, 224 × 224 or 2000 × 2000, is 256 image tokens,
a 1024 × 768 image 266 and a 96 × 48 image 253. Assign one image's features into another's run
only when the two give the same count.

## Image tokens attend to each other on the sliding blocks

12B's text config sets `use_bidirectional_attention` to `"vision"`: on a sliding block an image's
tokens attend to every token of the same image, before and after them, within the window; the
full blocks stay causal. The wrapper builds that mask from the processor's `mm_token_type_ids`,
so pass the whole encoding (`model.trace(prompt, images=[image])` does).

## The base checkpoint has no chat template

`google/gemma-4-12B` ships no chat template: put the processor's image token in the prompt
yourself, `f"{model.processor.image_token} What is this?"`. Audio enters through
`model.model.embed_audio`, another embedder without blocks, which keeps its native name.
""",
    },
}

#: Set by hues.py (lineage: Gemma).
PALETTE = {"hue": 165}
VLLM = False
QUIRKS = [
    "sandwich-norms", "scaled-residual-adds", "qk-norm", "sliding-window", "proportional-rotary", "scaled-embeddings",
    "softcapped-logits",
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
                "sliding_attention": "{sliding_window} window, head_dim {head_dim}",
                "full_attention": "full causal, wide heads, ¼ rotary",
            },
            "post_norm_note": "As on Gemma 2, 3 and 4, this norm follows the attention. On Llama the same name is the norm before the MLP.",
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
    "identity_note": "layer_scalar is layers[i]._module.layer_scalar, a per-block buffer. Exact in float32 on the pinned checkpoint.",
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "embed_tokens multiplies its lookup by √hidden_size, cast to the weight's dtype (61.97 on 12B, 62.0 in bfloat16), "
             "so token_embeddings is the scaled tensor and equals layers[0].input at every text position.",
    "layers": "Each block ends by multiplying its sum by layer_scalar (0.005 at block 11 on 12B), so a block's terms reach "
              "the last stream multiplied by every scalar from there on.",
    "norm": "Gemma4UnifiedRMSNorm multiplies by norm.weight itself, not 1 + weight.",
    "head": "lm_head shares its weight with embed_tokens on 12B (tie_word_embeddings).",
    "logits": "final_logit_softcapping is 30 on 12B: logits is 30 · tanh(lm_head.output / 30), and project_on_vocab applies the cap.",
}

NOTES = """
## The block, in order

```
a   = input_layernorm(x)
q   = rope(q_norm(q_proj(a)))         # per head
k   = rope(k_norm(k_proj(a)))         # per head
v   = v_norm(v_proj(a))               # per head, no gain; k_proj(a) on 12B's full blocks
h   = x + post_attention_layernorm(o_proj(attend(q, k, v)))      # no 1/√d
h   = h + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(h)))
out = h * layer_scalar
```

Four RMSNorms around the sublayers and three inside the attention. `post_attention_layernorm`
follows the attention; the MLP's input norm is `pre_feedforward_layernorm`. The block has no
per-layer branch and no mixture of experts: `per_layer_output` and the mixture's values are not
values of this family. `layer_scalar` is a buffer, `layers[i]._module.layer_scalar`, applied in
place.

## The block's sum is scaled; its terms are served unscaled

`attention_output` and `mlp_output` are the post-norms' outputs, before `layer_scalar`. The
identity is exact in float32 on the pinned checkpoint:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

scalar = model.layers[1]._module.layer_scalar
torch.testing.assert_close((x + attn + mlp) * scalar, out)
```

A vector added to `mlp_output` reaches `layer_output` multiplied by the block's scalar; one added
to `layer_output` does not. Block `i`'s terms reach the last stream multiplied by its own scalar
and every later one, so direct logit attribution weights each term by that product first. The
scalars are far from one on 12B: block 11's is 0.005.

## Queries, keys and values are normed, and the scores are not scaled

`q_norm` and `k_norm` norm each head before the rotary; `v_norm` norms each value head with no
gain, so every head of `attention_values` has RMS 1. The attention's `scaling` is 1.0: the scores
are `q · k` with no `1/√head_dim`, and the norms' gains set the temperature. `attention_queries`
and `attention_keys` are read after their norms and the rotary.

## Full blocks take their values from the key projection

12B sets `attention_k_eq_v`: its full blocks have no `v_proj`, and their values are
`v_norm(k_proj(a))`, the keys' projection before `k_norm` and the rotary. An edit to a full
block's `k_proj` output moves its keys and its values at once; `attention_keys` and
`attention_values` are served apart and edit apart. Each full block has one key/value head for
its 16 query heads, so an edit to a full block's `attention_keys` or `attention_values` reaches
all 16; a sliding block has 8, each serving two.

## Sliding and full blocks differ in width and rotary

`config.layer_types` makes every sixth block full on 12B (5, 11, …, 47), the last block among
them; the window is 1024 tokens. Full blocks have 512-wide heads and one key/value head; sliding
blocks 256-wide heads and 8. The root's sizes are the sliding blocks'; each block's own are on its
attention:

```python
model.head_dim, model.layers[5].self_attn.head_dim        # 256, 512 on 12B
model.num_kv_heads, model.layers[5].self_attn.num_kv_heads  # 8, 1 on 12B
```

Sliding blocks rotate all 256 dimensions with base 10,000. Full blocks use proportional rotary
(`partial_rotary_factor` 0.25, base 1,000,000): only a quarter of each head turns, at frequencies
spaced over the whole head; the other dimensions carry no position.

## The family serves borrowed keys and values; 12B borrows none

The attention is Gemma 4's: on a config with `num_kv_shared_layers` above 0 the last blocks attend
with an earlier block's keys and values, served as a copy private to the block. 12B sets
`num_kv_shared_layers` to 0, so every block computes its own. The pinned checkpoint sets 2: its
blocks 2 and 3 attend with blocks 0's and 1's keys.

## The readout is softcapped, and the norm gain is the weight

`logits` is `30 · tanh(lm_head.output / 30)` on 12B; `project_on_vocab` applies the final norm,
`lm_head` and the cap, so a lens at the last block reproduces `logits`. The vocabulary is 262,144
tokens and `lm_head` is `embed_tokens`' weight. `Gemma4UnifiedRMSNorm` multiplies by `weight`,
not `1 + weight`: folding the final norm into `lm_head` uses `model.norm._module.weight` as it is.

## Every checkpoint is a multimodal wrapper

The released repos are `gemma4_unified` checkpoints: the text-generation task builds
`Gemma4UnifiedForConditionalGeneration`, nnterp picks this family from `config.text_config`, and
the text stack sits under `model.model.language_model` with `lm_head` at the root.
`Gemma4UnifiedForCausalLM` is built only from a `gemma4_unified_text` config, as the suite's tiny
checkpoint is.
"""
