"""Gemma 1: Llama's block with Gemma's scaled embeddings, 1 + weight norms and a GeGLU MLP; PaliGemma wraps it."""

MODEL_TYPE = "gemma"
TITLE = "Gemma / PaliGemma"
SUBTITLE = (
    "Llama's pre-norm block with a GeGLU MLP: the embeddings are multiplied by √hidden_size before block 0, "
    "every RMSNorm multiplies by 1 + weight, and the head is the embedding matrix."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "google/gemma-2b"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "trl-internal-testing/tiny-GemmaForCausalLM"
CHECKPOINTS = [
    "google/gemma-2b", "google/gemma-2b-it",
    "google/gemma-7b", "google/gemma-7b-it",
    "google/gemma-1.1-2b-it", "google/gemma-1.1-7b-it",
    "google/codegemma-2b", "google/codegemma-7b",
    # Vision-language wrappers around a Gemma 1 text model (text_config.model_type gemma): model_type paligemma.
    # PaliGemma 2 wraps a Gemma 2 text model, so it is not this family.
    "google/paligemma-3b-pt-224", "google/paligemma-3b-pt-448", "google/paligemma-3b-pt-896",
    "google/paligemma-3b-mix-224", "google/paligemma-3b-mix-448",
]

#: The vision-language wrappers of this family, keyed by the wrapper's config.model_type. The vision encoder comes
#: from the checkpoint's vision_config.model_type (encyclopedia/vision/); what a config does not say is here.
WRAPPERS = {
    "paligemma": {
        "title": "PaliGemma",
        "pinned": "hf-tiny-v2/tiny-random-PaliGemmaForConditionalGeneration",
        "projector": "multi_modal_projector: one linear with a bias, from the vision encoder's width to the text "
                     "model's, over vision.tower_output",
        "projector_input": "`vision.tower_output`",
        "notes": """
## The projector is one linear over `tower_output`

`model.multi_modal_projector` (the `projector`) is a single `linear` with a bias, 1152 → 2048 on every
PaliGemma size. It reads the vision encoder's `last_hidden_state`, the patches after `post_layernorm`, so
`model.projector.input == model.vision.tower_output`, and a write to the last block or to `tower_output`
reaches the text model. `vision.image_features` is `model.projector.output` flattened over the images.
On the pinned tiny checkpoint:

```python
with model.trace(prompt, images=[image]):
    mask = model.vision.image_token_mask.save()
    out = model.vision.tower_output.save()
    fed = model.projector.input.save()
    projected = model.projector.output.save()
    features = model.vision.image_features.save()
    first = model.layers[0].input.save()

torch.equal(fed, out)                             # True: the projector reads tower_output
torch.equal(projected.flatten(0, 1), features)    # True
torch.equal(first[mask], features)                # True: the scatter
```

## One image token per patch

The projector maps each patch to one token, so an image is `(image_size // patch_size) ** 2` image
tokens: 256 on the 224 checkpoints (16 × 16 patches of 14 pixels), 1024 on the 448 ones and 4096 on
`paligemma-3b-pt-896` (the config's `num_image_tokens`). The processor resizes every image to the
checkpoint's `image_size`, so every image of a checkpoint gives the same count.

## The image comes first, unscaled

The processor puts the image tokens before the beginning-of-sequence token, then the prompt and a
newline: on the pinned tiny checkpoint positions 0 to 15 are the image, 16 is `<bos>`. The wrapper
scatters `model.projector.output` into the output of `embed_tokens` as it is, so the text rows of
`layers[0].input` carry the √hidden_size scale and the image rows carry none:
`layers[0].input[~mask] == token_embeddings[~mask]`. The wrapper numbers positions from 1
(`position_ids` is `arange(seq) + 1`), so a text token sits one rotary step later than on a
`GemmaForCausalLM`.

## A text-only trace takes the tokenizer's encoding

PaliGemma's processor refuses a prompt without an image. A text-only trace of the wrapper takes
the tokenizer's encoding instead; the mask is then all false and `vision.image_features` is never
reached.

```python
with model.trace(dict(model.tokenizer(text, return_tensors="pt"))):
    logits = model.logits.save()
```
""",
    },
}

#: Set by hues.py (lineage: Gemma).
PALETTE = {"hue": 153}
VLLM = True
QUIRKS = ["scaled-embeddings", "gain-norm"]

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
            "detail": "{num_heads} query, {num_kv_heads} key/value heads × {head_dim}",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "Named for its place on the stream, after the attention's add: it is the MLP's input norm, "
                             "as on Llama.",
            "contribution": "mlp_output",
            "detail": "GeGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "The embedding module multiplies its lookup by √hidden_size, cast to the weight's dtype (45.25 on 2B "
             "and 55.5 on 7B in bfloat16), so token_embeddings is the scaled tensor and equals layers[0].input on text.",
    "norm": "GemmaRMSNorm: the gain is 1 + norm.weight, not the stored weight.",
    "head": "lm_head shares its weight with embed_tokens (tie_word_embeddings). Nothing follows it: logits equals "
            "lm_head.output.",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))
out = h + mlp(post_attention_layernorm(h))
```

Two RMSNorms per block, each before its sublayer, none after: Llama's layout and Llama's names.
`post_attention_layernorm` is the MLP's input norm. The Gemma 2 and Gemma 3 blocks add a norm after
each sublayer and give this name to the norm after the attention, so a hook ported by name from
those families lands on a different tensor here.

## The contributions are the modules' outputs

`attention_output` is `self_attn.output[0]` and `mlp_output` is `mlp.output`, added to the stream
with nothing in between. On the pinned checkpoint, in float32:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

Scaling a module's output scales its contribution: `mlp.output[:] *= 0.5` and
`mlp.mlp_output[:] *= 0.5` are the same edit.

## Embeddings are scaled before block 0

`embed_tokens` multiplies its lookup by √hidden_size, cast to the weight's dtype first: 45.25 on 2B
and 55.5 on 7B in bfloat16, where the exact roots are 45.25 and 55.43. `token_embeddings` is the
scaled tensor and equals `layers[0].input` on text. The stored embedding weight is also the
unembedding (`tie_word_embeddings`), so a row of `embed_tokens.weight` is a token's unembedding
direction unscaled, and an edit to it changes the readout too.

## The norm gain is 1 + weight

`GemmaRMSNorm` normalizes in float32 and multiplies by `1 + weight`, then casts back. Folding a norm
into the next projection, or reading its gain, uses `1 + model.norm._module.weight`; the stored
weight is initialized at zero.

## Attention: multi-query on 2B, 256-wide heads

2B has 8 query heads over a single key/value head; 7B has 16 of each. `attention_keys` and
`attention_values` are read before `repeat_kv`, so on 2B they are `[batch, 1, seq, 256]` and an edit
to them reaches every query head. `head_dim` is 256 on both sizes, so 7B's 16 heads are 4096 wide
against a `hidden_size` of 3072, and the query scale is `head_dim ** -0.5`. There is no score
softcap, window or sink: the default `sdpa` load and `attn_implementation="eager"` run the same
attention, and eager serves the interior values.

## The activation follows the config

The MLP is `down_proj(act(gate_proj(x)) * up_proj(x))` with the config's `hidden_act`. Every
checkpoint runs the tanh approximation of GELU (`gelu_pytorch_tanh`) since transformers 5.18. The
`gemma-2b` and `gemma-7b` configs (and their `-it` and `codegemma-2b`) say `gelu`, a legacy value
meant as the tanh approximation, and since 5.18 `GemmaConfig` rewrites it to `gelu_pytorch_tanh`
when it loads (#49084), so `model.config.hidden_act` reads `gelu_pytorch_tanh`; up to 5.17 transformers
ran those checkpoints with the exact erf GELU. Gemma 1.1, `codegemma-7b` and PaliGemma's text config
give `gelu_pytorch_tanh` themselves.

## The readout

`model.logits` is `model.lm_head.output`: no cap and no scale, so `project_on_vocab` is
`lm_head(norm(hidden))` with the `1 + weight` gain inside `norm`, and a logit lens at the last block
equals `logits`.

## What this family module covers

Every checkpoint whose config says `model_type` `gemma`: Gemma 2B and 7B with their instruction-tuned
`-it` and 1.1 releases, CodeGemma 2B and 7B, and the text model inside PaliGemma 1. PaliGemma 2
wraps a Gemma 2 text model and is the `gemma2` family's.
"""
