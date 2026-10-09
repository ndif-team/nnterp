"""Ministral 3: the text model of the Ministral 3 vision-language checkpoints, Llama's block with position-scaled queries."""

MODEL_TYPE = "ministral3"
TITLE = "Ministral 3"
SUBTITLE = (
    "Llama's pre-norm block whose queries are multiplied, after the rotary embedding, by a factor that grows "
    "with the position past 16384 tokens; every checkpoint is a vision-language model on the Pixtral encoder."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "mistralai/Ministral-3-3B-Base-2512"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "yujiepan/ministral-3-tiny-random"
#: Every checkpoint is a `mistral3` wrapper whose text_config is `ministral3`.
CHECKPOINTS = [
    "mistralai/Ministral-3-3B-Base-2512", "mistralai/Ministral-3-3B-Instruct-2512",
    "mistralai/Ministral-3-8B-Base-2512", "mistralai/Ministral-3-8B-Instruct-2512",
    "mistralai/Ministral-3-8B-Reasoning-2512",
    "mistralai/Ministral-3-14B-Base-2512", "mistralai/Ministral-3-14B-Instruct-2512",
]

#: The vision-language wrapper every checkpoint above is, keyed by its config.model_type. The vision encoder comes from
#: the checkpoint's vision_config.model_type (encyclopedia/vision/pixtral.py); what a config does not say is here.
WRAPPERS = {
    "mistral3": {
        "title": "Mistral 3",
        "pinned": "yujiepan/mistral-3-tiny-random",
        "projector": "an RMSNorm on each patch, patch_merger folding each 2 × 2 block of an image's patches into one "
                     "(concatenated, then merging_layer, a linear), then linear_1, GELU, linear_2",
        "projector_input": "`vision.tower_output[0]`",
        "quirks": ["pooled-projector"],
        "notes": """
## The projector merges each 2 × 2 block of patches

`model.multi_modal_projector` (the `projector`) reads the last block's stream with its leading 1
dropped: `model.projector.input` is `vision.tower_output[0]`, `[patches, vision_hidden]`
(`vision_feature_layer` is `-1`). It norms each patch (`norm`, an RMSNorm), then `patch_merger`
cuts each image's grid into 2 × 2 blocks (`spatial_merge_size`), concatenates each block's four
patches into one `4 * vision_hidden` vector and maps it back to `vision_hidden` with
`merging_layer`, a linear without bias; `linear_1`, GELU and `linear_2` take it to the text model's
width. So an image of `rows × columns` patches is `(rows // 2) * (columns // 2)` image tokens, and
`vision.image_features` is `model.projector.output` as it is, `[image_tokens, hidden]`.

```python
with model.trace(prompt, images=[square, wide]):
    mask = model.vision.image_token_mask.save()
    tower = model.vision.tower_output.save()    # read before the projector
    fed = model.projector.input.save()
    features = model.vision.image_features.save()
    first = model.layers[0].input.save()

torch.equal(fed, tower[0])                      # True
torch.equal(first[mask], features)              # True
```

## Row breaks between the image tokens

The processor writes an image as one `[IMG]` per image token, row by row, with `[IMG_BREAK]` after
every row but the last and `[IMG_END]` after the last. Only `[IMG]` is in
`vision.image_token_mask`; the break and end tokens keep their own embeddings. On the pinned tiny
checkpoint an 84 × 84 and a 56 × 112 image at 14-pixel patches are 36 and 32 patches and 9 and 8
`[IMG]` tokens, with 3 `[IMG_BREAK]` and 2 `[IMG_END]`. On every size the vision encoder is 24
blocks of width 1024 with 14-pixel patches, so an image token covers 28 × 28 pixels, and the
processor resizes an image's longest side to at most 1540 pixels (`longest_edge`).
""",
    },
}

#: Set by hues.py (lineage: Llama and its kin).
PALETTE = {"hue": 269}
VLLM = False
QUIRKS: list[str] = []

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
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} query, {num_kv_heads} key/value heads",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "Named for its place on the stream, after the attention's add: it is the MLP's input norm, "
                             "and its input is the stream between the two adds.",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "A plain lookup: no scale and no position embedding is added, so token_embeddings equals "
             "layers[0].input on a text prompt. Positions enter as a rotation of queries and keys, and a scale "
             "of the queries, inside each attention.",
    "norm": "RMSNorm whose gain is norm.weight as stored.",
    "head": "Tied to embed_tokens on 3B (tie_word_embeddings), untied on 8B and 14B. Nothing follows it: logits "
            "equals lm_head.output, and project_on_vocab is lm_head(norm(hidden)).",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))
out = h + mlp(post_attention_layernorm(h))
```

Llama's block: two RMSNorms per block, each before its sublayer, none after, and a SwiGLU MLP.
`attention_output` is the attention module's output and `mlp_output` the MLP's, added to the stream
with nothing in between, so the identity holds exactly (float32, pinned checkpoint):

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

## `attention_queries` are the position-scaled queries

After the rotary embedding the attention multiplies each query by
`1 + beta * log(1 + floor(position / original_max_position_embeddings))`, with `beta` the
`rope_parameters`' `llama_4_scaling_beta` (`0.1`) and `original_max_position_embeddings` `16384` on
every size. `attention_queries` is read at the attention function's arguments, after the scale. The
factor is exactly 1 below position 16384, so on shorter prompts the queries are the rotated ones;
it is about 1.069 from 16384 and about 1.22 from 131072. The rotated, unscaled queries are the
output of the source's `apply_rotary_pos_emb_0`:

```python
attn = model.layers[1].self_attn
with model.trace(prompt):
    rotated = attn.source.apply_rotary_pos_emb_0.output[0].save()
    q = attn.attention_queries.save()            # rotated * the position factor
```

On the pinned checkpoint rewritten with `beta` 0.1 and `original_max_position_embeddings` 4,
`q` equals `rotated` times the factor at every position, and `rotated` exactly at positions 0 to 3.
The keys are not scaled. The rotary itself is YaRN (`rope_type` `yarn`, `factor` 16) over a
`max_position_embeddings` of 262144; its `attention_scaling` is 1.0 on 3B and 8B, so the cosines
and sines are not rescaled.

## Grouped-query attention, as wide as the heads make it

Every size has 32 query heads over 8 key/value heads of width 128: query head `h` reads key/value
head `h // 4`, and `attention_keys` and `attention_values` are read before `repeat_kv`,
`[batch, 8, seq, 128]`. The heads are 4096 wide together, which is wider than the 3B stream (3072),
equal to 8B's (4096) and narrower than 14B's (5120): `q_proj` and `o_proj` change width on 3B and
14B, and `attention_head_outputs` flattened is 4096 wide on every size.

## Every checkpoint is a vision-language wrapper

`Ministral-3-*-2512` checkpoints are `model_type` `mistral3` whose `text_config` is `ministral3`,
so each loads with `task="image-text-to-text"`, and `model.layers` are the wrapper's
`model.language_model.layers`. A text-only trace runs on the wrapper as on a text model; the image
values are served when the trace is given an image.

## Target tokens

The tokenizer is the 131072-token Tekken vocabulary: `" Paris"` is `ĠParis` (`6993`) and `"Paris"`
is `Paris` (`42572`), two different tokens (checked on 3B). Pick the one the prompt's spacing
produces, and assert it is one token:

```python
ids = model.tokenizer(" Paris", add_special_tokens=False).input_ids
assert len(ids) == 1
```
"""
