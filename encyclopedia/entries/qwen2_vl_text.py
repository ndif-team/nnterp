"""Qwen2-VL: the text model of the Qwen2-VL checkpoints, Qwen2's block under multimodal rotary embeddings."""

MODEL_TYPE = "qwen2_vl_text"
TITLE = "Qwen2-VL"
SUBTITLE = (
    "Qwen2's block, biased q, k and v, with multimodal rotary embeddings: three position streams (time, row, column) "
    "folded into one cos/sin before block 0, so an image's tokens share positions by row and column."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "Qwen/Qwen2-VL-2B-Instruct"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "yujiepan/qwen2-vl-tiny-random"
#: Every checkpoint is the vision-language wrapper (model_type qwen2_vl): transformers has no causal-LM class for it.
CHECKPOINTS = [
    "Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-2B",
    "Qwen/Qwen2-VL-7B-Instruct", "Qwen/Qwen2-VL-7B",
    "Qwen/Qwen2-VL-72B-Instruct", "Qwen/Qwen2-VL-72B",
]

#: The vision-language wrapper every checkpoint above is, keyed by its config.model_type. The vision encoder comes from
#: the checkpoint's vision_config.model_type (encyclopedia/vision/qwen_vit.py); what a config does not say is here.
WRAPPERS = {
    "qwen2_vl": {
        "title": "Qwen2-VL",
        "pinned": "yujiepan/qwen2-vl-tiny-random",
        "projector": "merger, inside the vision encoder: a LayerNorm (ln_q) on each patch, then an MLP (Linear, GELU, "
                     "Linear) over each 2 × 2 block of patches concatenated, one image token per block",
        "projector_input": "the merger's input: the last block's output",
        "notes": """
## The merger folds four patches into one image token

`model.projector` is `model.visual.merger`, the vision encoder's last module: `ln_q` (a LayerNorm
over `vision_hidden`) on each patch, then each 2 × 2 block of consecutive patches concatenated into
one `4 * vision_hidden` vector, and `mlp`: `Linear`, GELU, `Linear` to the text model's width. On
every checkpoint the vision encoder is 1280 wide, so the merger maps 5120 to 5120 and then to 1536
(2B), 3584 (7B) or 8192 (72B). A patch is 14 pixels, so an image token covers 28 × 28 pixels, and an
image of `t * h * w` patches (its `image_grid_thw` row) is `t * h * w / 4` image tokens: 256 for a
448 × 448 image. `vision.image_features` is `model.projector.output` as it is, `[image_tokens, hidden]`.

```python
with model.trace(prompt, images=[image]):
    mask = model.vision.image_token_mask.save()
    fed = model.projector.input.save()          # [patches, vision_hidden]
    merged = model.projector.output.save()
    features = model.vision.image_features.save()
    first = model.layers[0].input.save()

fed.shape[0] // 4 == mask.sum()                 # True: one image token per 2 x 2 block
torch.equal(merged, features)                   # True
torch.equal(first[mask], features)              # True
```

## The image tokens sit between vision markers

The processor expands each `<|image_pad|>` (151655) in the prompt to the image's token count, between
`<|vision_start|>` and `<|vision_end|>`; `vision.image_token_mask` marks the pads only. The Instruct
chat template writes the three for an `{"type": "image"}` entry. The base checkpoints' template takes
one content list, with no roles, and writes the same markers:

```python
content = [{"type": "image"}, {"type": "text", "text": "A photo of"}]
prompt = model.processor.apply_chat_template(content)
# '<|vision_start|><|image_pad|><|vision_end|>A photo of'
```
""",
    },
}

#: The Qwen lineage sits at 285 (qwen2), 297 (qwen3) and 273 (qwen3_5_text); this family takes 279.
PALETTE = {"hue": 279}
VLLM = False
QUIRKS = ["qkv-bias"]

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
            "detail": "{num_heads} heads, {num_kv_heads} kv; M-RoPE, qkv bias",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}, no biases",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "An unscaled lookup: at the text positions token_embeddings equals layers[0].input. The tokenizer "
             "prepends no BOS.",
    "layers": "Every block receives the same cos/sin, which the text model's rotary_emb folds from the three "
              "M-RoPE position streams before block 0.",
    "head": "lm_head shares its weight with embed_tokens on 2B (tie_word_embeddings); 7B and 72B have their own. "
            "logits is lm_head.output, with no softcap or scale.",
}

NOTES = """
## The block, in order

The text block is Qwen2's: a pre-norm attention whose `q_proj`, `k_proj` and `v_proj` add a bias
(`o_proj` has none), then a SwiGLU MLP with no biases, each behind an RMSNorm.

```
a   = input_layernorm(x)
h   = x + o_proj(attn(rope(q_proj(a)), rope(k_proj(a)), v_proj(a)))
out = h + mlp(post_attention_layernorm(h))     # SwiGLU, no biases
```

The block returns a tensor and adds both sublayers' outputs to the stream, so the identity is the
plain sum:

```python
with model.trace(prompt, images=[image]):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

## Positions are three streams

The wrapper passes the text model `position_ids` of `[3, batch, seq]`: a temporal, a height and a
width position per token. On text the three are equal and count up. An image's tokens all take the temporal
position the image starts at, and each takes its row and its column in the image's grid of merged
tokens as the other two, offset by the same start. The text after an image resumes one past the
image's largest position, so an image `h` tokens high and `w` wide advances the positions by
`max(h, w)`, not by `h * w`: after an image, a token's position is less than its index.

```python
rotary = model.model.language_model.rotary_emb
with model.trace(prompt, images=[image]):
    mask = model.vision.image_token_mask.save()
    positions = rotary.inputs[0][1].save()          # [3, batch, seq]: time, height, width

positions[:, 0, mask[0]]                            # the image tokens' three positions
```

`rotary_emb` folds the three into one `cos`/`sin` by `mrope_section`, `[16, 24, 24]` on every
checkpoint: of each head's 64 rotary frequencies, the first 16 turn with the temporal position, the
next 24 with the height, the last 24 with the width, so in a 128-wide head dimensions 0 to 15 and 64
to 79 carry time, 16 to 39 and 80 to 103 the row, 40 to 63 and 104 to 127 the column. The attention
applies that pair with `rotate_half` before the attention interface, so `attention_queries` and
`attention_keys` are the rotated queries and keys, and on a prompt with no image the rotation is the
ordinary rotary one. `rope_theta` is 10⁶.

## Grouped-query attention

2B has 12 query heads over 2 key/value heads, 7B 28 over 4, 72B 64 over 8; `head_dim` is 128 on
all three. An edit to key/value head `j` on 2B reaches query heads `6j` to `6j + 5`. The score
scale is `head_dim ** -0.5`; there is no window and no softcap. `attn_implementation="eager"` is
needed for `attention_scores` and `attention_probabilities` on the text blocks; on the vision
encoder's blocks those two are unavailable under every implementation.

## The image enters at the scatter

The wrapper writes `vision.image_features` into the token embeddings at the image tokens before
block 0, so `token_embeddings` holds the `<|image_pad|>` embedding there and `layers[0].input` the
features: they agree at the text positions only. `layers[0].input[vision.image_token_mask] ==
vision.image_features` holds exactly. To ablate or patch the image as the text model receives it,
edit `vision.image_features`:

```python
with model.trace(prompt, images=[image]):
    model.vision.image_features[:] = 0
    ablated = model.logits.save()
```

## The readout is the plain projection

`model.logits` equals `model.lm_head.output`. On 2B the embedding and the unembedding are one
matrix (`tie_word_embeddings`), so an edit to `embed_tokens.weight` edits `lm_head`; 7B and 72B have
their own.

## The chat template writes a system turn

The tokenizer prepends no BOS (`bos_token` is `None`). Given no system message, the Instruct
template writes `You are a helpful assistant.`, so a templated prompt opens with a system turn and
position 0 is `<|im_start|>`. The generation configs end on `<|im_end|>` (151645) or
`<|endoftext|>` (151643).

## What loads as this family

Every checkpoint here is `Qwen2VLForConditionalGeneration` (`model_type` `qwen2_vl`) with a
`text_config` of `model_type` `qwen2_vl_text`. transformers registers no causal-LM class for it, so
every load is the wrapper with its processor, `task="image-text-to-text"`: the text model at
`model.language_model`, the vision encoder at `model.visual` as `model.vision`. The text model's
names and values are the same whether or not a prompt carries an image.
"""
