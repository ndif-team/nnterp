"""Mistral: Llama's block, with a sliding window on the first 7B and two vision encoders on its wrappers."""

MODEL_TYPE = "mistral"
TITLE = "Mistral / Mistral Nemo / Mistral Small"
SUBTITLE = (
    "Llama's pre-norm block with grouped-query attention; on Mistral-7B-v0.1 every block attends over a "
    "sliding window of the last 4096 positions, and from Nemo on the attention is narrower than the stream."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "mistralai/Mistral-7B-v0.1"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-MistralForCausalLM"
CHECKPOINTS = [
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-v0.3", "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-Nemo-Base-2407",
    # Vision-language wrappers around a Mistral text model, each a key of WRAPPERS by its config's model_type.
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503",  # mistral3, Pixtral
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",  # mistral3, Pixtral
    "mistral-community/pixtral-12b",                  # llava, Pixtral
    "llava-hf/llava-v1.6-mistral-7b-hf",              # llava_next, CLIP
    "llava-hf/bakLlava-v1-hf",                        # llava, CLIP
]

#: The vision-language wrappers of this family, keyed by the wrapper's config.model_type. The vision encoder comes
#: from the checkpoint's vision_config.model_type (encyclopedia/vision/pixtral.py, clip.py); what a config does not
#: say is here. The `llava` key holds two checkpoints on two vision encoders (Pixtral-12B and BakLLaVA), so its
#: record states what each reads.
WRAPPERS = {
    "mistral3": {
        "title": "Mistral 3",
        "pinned": "hf-tiny-v2/tiny-random-Mistral3ForConditionalGeneration",
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
checkpoint a 36 × 36 and a 24 × 36 image at 6-pixel patches are 9 and 6 `[IMG]` tokens (3 × 3 and
2 × 3 merged blocks), with 3 `[IMG_BREAK]` and 2 `[IMG_END]`. On Mistral Small 3.1 and 3.2 a patch
is 14 pixels, so an image token covers 28 × 28 pixels, and the processor resizes an image's longest
side to at most 1540 pixels (`longest_edge`).
""",
    },
    "llava": {
        "title": "Llava (Pixtral-12B, BakLLaVA)",
        "pinned": "mistral-community/pixtral-12b",
        "projector": "a two-layer MLP (linear_1, GELU, linear_2), one image token per patch, over the block "
                     "vision_feature_layer names: the last on Pixtral-12B, block -2 without the CLS token on BakLLaVA",
        "projector_input": "`vision.tower_output` (Pixtral-12B); `vision.layers[-2].layer_output[:, 1:]` (BakLLaVA)",
        "notes": """
## Two checkpoints, two vision encoders, one wrapper

`mistral-community/pixtral-12b` and `llava-hf/bakLlava-v1-hf` are both `model_type` `llava`
around a Mistral text model: Pixtral-12B on the Pixtral vision encoder, BakLLaVA on CLIP. The
wrapper's projector is the same two-layer MLP (`linear_1`, GELU, `linear_2`), one image token per
patch, and `vision.image_features` is `model.projector.output` flattened over the images. What
feeds it is the block `vision_feature_layer` names, sliced by
`vision_feature_select_strategy`, and the two configs differ:

- Pixtral-12B: `-1` and `"full"`, so `model.projector.input` is `vision.tower_output`,
  `[1, patches, vision_hidden]`, every image's patches packed. A patch is 16 pixels and the
  processor resizes an image's longest side to at most 1024, so an image is up to 64 × 64 image
  tokens, written row by row with `[IMG_BREAK]` after every row but the last and `[IMG_END]` after
  the last; only `[IMG]` is in `vision.image_token_mask`.
- BakLLaVA: `-2` and `"default"`, so `model.projector.input` is
  `vision.layers[-2].layer_output[:, 1:]`, block -2 without its CLS token: the last block and
  `vision.tower_output` are computed and discarded. An image is 576 image tokens (24 × 24 patches of
  14 pixels at 336).

```python
with model.trace(prompt, images=[image]):
    stream = model.vision.layers[-2].layer_output.save()
    tower = model.vision.tower_output.save()
    fed = model.projector.input.save()

torch.equal(fed, tower)               # True on Pixtral-12B
torch.equal(fed, stream[:, 1:])       # True on BakLLaVA
```

""",
    },
    "llava_next": {
        "title": "LLaVA-NeXT",
        "pinned": "trl-internal-testing/tiny-LlavaNextForConditionalGeneration",
        "projector": "a two-layer MLP over each crop's block -2 patches, the CLS token dropped; the wrapper unpads its "
                     "output and adds newline tokens",
        "projector_input": "each crop's `vision.layers[-2].layer_output[:, 1:]`",
        "quirks": ["tiled-images", "unpadded-features"],
        "notes": """
## Crops as rows, and features that are not the projector's output

The processor cuts an image into a base image and crops at a resolution from
`image_grid_pinpoints`, each a row of the vision encoder's batch, and the projector reads block -2
without the CLS token (`vision_feature_layer` is `-2` on `llava-v1.6-mistral-7b-hf`):
`model.projector.input` is `vision.layers[-2].layer_output[:, 1:]`. The wrapper then unpads the
projector's output to the image's aspect ratio and appends `image_newline` after each row of
patches, so `vision.image_features` has another row count than `model.projector.output`. On the
pinned tiny checkpoint a 64 × 64 image is three rows of 576 patches, the projector returns
`[3, 576, hidden]`, and `vision.image_features` is 1176 rows, 24 of them `image_newline`. Edit
`vision.image_features` for what the text model receives.
""",
    },
}

#: The Llama lineage sits at the hash of `llama` (148); Gemma's 133, 145 and 157 are near, so this family takes 152.
PALETTE = {"hue": 152}
VLLM = True
QUIRKS: list[str] = []

#: What the visualization draws: the sublayers in forward order, each with its norms,
#: the standard value that is its contribution, and the values read inside it.
#: ``detail`` is formatted with the sizes and the config keys.
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
             "layers[0].input. Positions enter as a rotation of queries and keys inside each attention.",
    "norm": "RMSNorm whose gain is norm.weight as stored.",
    "head": "Untied from embed_tokens on every checkpoint listed here. Nothing follows it: logits equals "
            "lm_head.output, and project_on_vocab is lm_head(norm(hidden)).",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))
out = h + mlp(post_attention_layernorm(h))
```

Llama's block: two RMSNorms per block, each before its sublayer, none after, and a SwiGLU MLP
(`gate_proj`, `up_proj`, `down_proj`, SiLU). `post_attention_layernorm` is named for where it sits
on the stream, after the attention's add: it is the MLP's input norm.

## The contributions are the modules' outputs

`attention_output` is the attention module's output and `mlp_output` the MLP's, added to the
stream with nothing in between, so the identity holds exactly (float32, pinned checkpoint):

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

## A sliding window on Mistral-7B-v0.1

`config.sliding_window` is `4096` on Mistral-7B-v0.1 and BakLLaVA and `None` on
v0.3, Nemo, Small 3.1 and 3.2, Pixtral-12B and `llava-v1.6-mistral`. The config has no
`layer_types`: where the window is set, every block takes it. A query attends to its own position
and the 4095 before it, so on a prompt shorter than the window the mask is the plain causal mask.
The window is in the mask the attention receives, so under `attn_implementation="eager"`
`attention_probabilities` is exactly zero beyond it (checked on the pinned checkpoint loaded with
`sliding_window=3`: every entry three or more positions back is `0.0`).

```python
model.config.get_text_config().sliding_window   # 4096 on v0.1, None on v0.3
```

## Grouped-query attention, narrower than the stream from Nemo on

Every checkpoint listed has 32 query heads over 8 key/value heads of width 128.
`attention_keys` and `attention_values` are read before `repeat_kv`, `[batch, 8, seq, 128]`, and
query head `h` reads key/value head `h // 4`: an edit to key/value head `j` reaches query heads
`4j` to `4j + 3`. On the 7B checkpoints 32 × 128 is the stream's width, 4096. On Nemo, Small 3.1
and 3.2 and Pixtral-12B the stream is 5120 wide and the heads still 32 × 128: `q_proj` maps 5120
to 4096 and `o_proj` 4096 back to 5120, so `attention_head_outputs` flattened is 4096 wide, not
`hidden_size`.

## Target tokens differ by tokenizer

Mistral-7B v0.1 and v0.3 use a SentencePiece vocabulary (32000 and 32768 tokens): `" Paris"` and
`"Paris"` both encode to `▁Paris` (`5465` on v0.1, `6233` on v0.3). Nemo, Small 3.1 and Pixtral-12B
use the 131072-token Tekken vocabulary, where they are two tokens: `" Paris"` is `ĠParis` (`6993`)
and `"Paris"` is `Paris` (`42572`). The pinned tiny checkpoint's tokenizer encodes `" Paris"` to
two tokens, `['▁', '▁Paris']`. Decode the target ids on the checkpoint you run.

```python
ids = model.tokenizer(" Paris", add_special_tokens=False).input_ids
assert len(ids) == 1, model.tokenizer.convert_ids_to_tokens(ids)
```

## The readout and the embeddings

`model.logits` is `model.lm_head.output`: no cap and no scale, so `project_on_vocab` is
`lm_head(norm(hidden))`. `token_embeddings` is `layers[0].input` on a text prompt (checked on the
pinned checkpoint). No checkpoint listed ties `lm_head` to `embed_tokens`.

## What this family module covers

The module serves every checkpoint whose config resolves to `model_type` `mistral`: the 7B base
and instruct models v0.1 to v0.3, Mistral Nemo, Mistral Small 24B, and the text model of four
vision-language wrappers (Mistral Small 3.1 and 3.2, Pixtral-12B, `llava-v1.6-mistral`,
BakLLaVA). `mistralai/Ministral-8B-Instruct-2410`'s `config.json` says `mistral`, but
`AutoConfig` reads a `mistral` config that has `layer_types` as `ministral`, so it is the
`ministral` family. Mixtral is
the `mixtral` family, and Ministral 3's text model `ministral3`.
"""
