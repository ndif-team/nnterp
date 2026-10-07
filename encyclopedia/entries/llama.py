"""Llama: the block the standard names are taken from."""

MODEL_TYPE = "llama"
TITLE = "Llama"
SUBTITLE = (
    "The reference block: one RMSNorm before each sublayer and none after, so what reaches the "
    "residual stream is the attention's and the MLP's own output, with rotary positions applied to "
    "queries and keys inside the attention."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "meta-llama/Llama-3.1-8B"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-LlamaForCausalLM"
CHECKPOINTS = [
    "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-70b-hf",
    "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-70B",
    "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B", "meta-llama/Llama-3.1-405B",
    "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    # Vision-language wrappers around a Llama text model, each a key of WRAPPERS by its config's model_type.
    "llava-hf/llava-1.5-7b-hf", "llava-hf/llava-1.5-13b-hf",
    "llava-hf/vip-llava-7b-hf",
    "llava-hf/llava-v1.6-vicuna-7b-hf",
    "deepseek-community/deepseek-vl-1.3b-chat",
    "HuggingFaceM4/Idefics3-8B-Llama3",
    "HuggingFaceTB/SmolVLM-Instruct",  # model_type idefics3
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct",  # model_type smolvlm
]

#: The vision-language wrappers of this family, keyed by the wrapper's config.model_type. The vision encoder comes
#: from the checkpoint's vision_config.model_type (encyclopedia/vision/); what a config does not say is here.
WRAPPERS = {
    "llava": {
        "title": "Llava 1.5",
        "pinned": "trl-internal-testing/tiny-LlavaForConditionalGeneration",
        "projector": "a two-layer MLP (linear_1, GELU, linear_2) over block -2's patches, the CLS token dropped",
        "notes": """
## The projector reads block -2, without the CLS token

`vision_feature_layer` is `-2` and `vision_feature_select_strategy` is `"default"`: the projector
receives `vision.layers[-2].layer_output[:, 1:]`, so the last block and `vision.tower_output` are
computed and discarded. On `llava-hf/llava-1.5-7b-hf`, asked the colour of a red square, zeroing
`vision.tower_output` leaves `Red` at 0.990, and zeroing `vision.layers[-2].layer_output` drops it to
0.001.

```python
with model.trace(prompt, images=[image]):
    stream = model.vision.layers[-2].layer_output.save()
    fed = model.projector.input.save()

torch.equal(fed, stream[:, 1:])   # True
```

## One token per patch

The projector maps each patch to one token, so an image is 576 image tokens on 7B and 13B (24 × 24
patches of 14 pixels at 336), and `vision.image_features` is `model.projector.output` flattened over
the images: `[576, hidden_size]` per image.
""",
    },
    "vipllava": {
        "title": "VipLlava",
        "pinned": "hf-tiny-v2/tiny-random-VipLlavaForConditionalGeneration",
        "projector": "a LayerNorm and a two-layer MLP over five blocks' patches concatenated, the CLS token dropped from each",
        "notes": """
## The projector reads five blocks

`vision_feature_layers` is `[-2, -5, -8, -11, 6]` on `vip-llava-7b-hf`, indices into the vision encoder's
hidden states, where `0` is the stream entering block 0 and `k > 0` is block `k - 1`'s
`layer_output`. So the projector reads `vision.layers[i].layer_output` of blocks 22, 19, 16, 13 and 5,
each without its CLS token, concatenated on the last axis: `projector.input` is
`[images, 576, 5120]`, and `projector_layernorm` norms it before the MLP. The last block and
`vision.tower_output` are computed and discarded. `vision.image_features` is
`model.projector.output` flattened over the images.

```python
streams = []
with model.trace(prompt, images=[image]):
    for k in (5, 13, 16, 19, 22):
        streams.append(model.vision.layers[k].layer_output.save())
    fed = model.projector.input.save()

order = [4, 3, 2, 1, 0]   # the config's order: blocks 22, 19, 16, 13, 5
torch.equal(fed, torch.cat([streams[j][:, 1:] for j in order], -1))   # True
```
""",
    },
    "llava_next": {
        "title": "LLaVA-NeXT",
        "pinned": "hf-tiny-v2/tiny-random-LlavaNextForConditionalGeneration",
        "projector": "a two-layer MLP over each crop's block -2 patches, the CLS token dropped; the wrapper unpads its output and adds newline tokens",
        "quirks": ["tiled-images", "unpadded-features"],
        "notes": """
## Crops as rows, and features that are not the projector's output

The processor cuts an image into a base image and crops at a resolution from
`image_grid_pinpoints`, each a row of the vision encoder's batch, and the projector reads block -2 without
the CLS token as on Llava 1.5 (`vision_feature_layer` is `-2` on `llava-v1.6-vicuna-7b-hf`). The
wrapper then unpads the projector's output to the image's aspect ratio and appends
`image_newline` after each row of patches, so `vision.image_features` has another row count than
`model.projector.output`. On the pinned tiny checkpoint the projector returns 8 rows (the base image
and one crop, 4 patches each) and `vision.image_features` 10, 2 of them `image_newline`. Edit `vision.image_features` for what
the text model receives.
""",
    },
    "deepseek_vl": {
        "title": "DeepSeek-VL",
        "pinned": "hf-tiny-v2/tiny-random-DeepseekVLForConditionalGeneration",
        "projector": "aligner, a two-layer MLP (linear1, GELU, linear2) over vision.tower_output",
        "notes": """
## The aligner reads `tower_output`

`model.aligner` is the `projector`: `linear1`, GELU, `linear2`, applied to `vision.tower_output`, the
patches after `post_layernorm`. So `model.projector.input == model.vision.tower_output`, a write to the
last block reaches the text model through the norm, and `vision.image_features` is
`model.projector.output` flattened. An image is 576 image tokens on 1.3B (24 × 24 patches of 16 pixels
at 384), the processor's `num_image_tokens`.
""",
    },
    "idefics3": {
        "title": "Idefics 3",
        "pinned": "trl-internal-testing/tiny-Idefics3ForConditionalGeneration",
        "projector": "connector: a pixel shuffle folding each scale_factor × scale_factor block of patches into one token, then modality_projection, a linear",
        "quirks": ["tiled-images", "pooled-projector"],
        "notes": """
## Tiles as rows, pixel-shuffled into tokens

The processor splits an image into tiles of `image_size` pixels and appends the whole image resized
to one tile, each a row of the vision encoder's batch. `model.connector` (the `projector`) reads
`vision.tower_output`, folds each `scale_factor` × `scale_factor` block of neighbouring patches into one
token, their widths concatenated, and projects it with `modality_projection`, a linear without bias.
On `Idefics3-8B-Llama3` (`scale_factor` 2) a 364-pixel tile's 26 × 26 = 676 patches become 169 tokens;
`HuggingFaceTB/SmolVLM-Instruct` is an `idefics3` checkpoint with `scale_factor` 3, so its 384-pixel
tile's 729 patches become 81.

## The features go in through `inputs_merger`

`vision.image_features` is read at `inputs_merger`'s `image_hidden_states` argument, which is
`model.projector.output` flattened, tiles in order:
`layers[0].input[vision.image_token_mask] == vision.image_features` holds as on every wrapper.
""",
    },
    "smolvlm": {
        "title": "SmolVLM",
        "pinned": "trl-internal-testing/tiny-SmolVLMForConditionalGeneration",
        "projector": "connector: a pixel shuffle folding each scale_factor × scale_factor block of patches into one token, then modality_projection, a linear",
        "quirks": ["tiled-images", "pooled-projector"],
        "notes": """
## Idefics 3's layout, with a 3 × 3 shuffle

SmolVLM keeps Idefics 3's vision encoder, connector and merge: tiles as rows of the vision encoder's batch, the
connector reading `vision.tower_output`, and `vision.image_features` read at `inputs_merger`'s
`image_hidden_states`, `model.projector.output` flattened. On `SmolVLM2-2.2B-Instruct` the
`scale_factor` is 3, so a 384-pixel tile's 27 × 27 = 729 patches become 81 tokens.
""",
    },
}

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
    "head": "Untied on 8B, 70B and Llama-2-7B; tied to embed_tokens on Llama-3.2-1B and 3B (tie_word_embeddings). "
            "Nothing follows it: logits equals lm_head.output, and project_on_vocab is lm_head(norm(hidden)).",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))
out = h + mlp(post_attention_layernorm(h))
```

Two RMSNorms per block, each before its sublayer, none after. `post_attention_layernorm` is
named for where it sits on the stream, after the attention's add: it is the MLP's input norm.
Some families with a norm after each sublayer give the name to a different module (on Gemma-2
it is the norm after the attention), so check the block's forward before porting a hook by name.

## The contributions are the modules' outputs

`attention_output` is `self_attn.output[0]` and `mlp_output` is `mlp.output`: the same tensors,
added to the stream with nothing in between. The identity holds exactly, in float32 on the
pinned checkpoint and in bfloat16 on Llama-3.2-1B:

```python
with model.trace(prompt):
    x = model.layers[5].input.save()
    attn = model.layers[5].self_attn.attention_output.save()
    mlp = model.layers[5].mlp.mlp_output.save()
    out = model.layers[5].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

So scaling a module's output scales its contribution: `model.layers[i].mlp.output[:] *= 0.5`
and `model.layers[i].mlp.mlp_output[:] *= 0.5` give the same logits. The inputs are the norms'
outputs: `self_attn.input` is `input_layernorm.output`, `mlp.input` is
`post_attention_layernorm.output`. The stream between the two adds has no standard value; it is
`model.layers[i].post_attention_layernorm.input`, which equals
`model.layers[i].input + model.layers[i].self_attn.attention_output`.

## The default load runs the same attention as eager

A default load runs `sdpa`, and the six attention interior values report unavailable;
`attn_implementation="eager"` serves them. Llama has no score softcap, window or sink, so the two
paths compute the same function: on Llama-3.2-1B in float32 their logits differ by at most
`2e-5`, and in bfloat16 by up to `0.25` with the same top token at every position of a test prompt. The query
scale is `head_dim ** -0.5` (`128 ** -0.5` on 8B, `64 ** -0.5` on 3.2-1B).

## Grouped-query attention on the 3.x sizes

8B and 3.2-1B have 32 query heads over 8 key/value heads, 3.2-3B 24 over 8, 70B 64 over 8.
`attention_keys` and `attention_values` are read before `repeat_kv`, `[batch, 8, seq, head_dim]`,
and query head `h` reads key/value head `h // 4` on 8B: an edit to key/value head `j` reaches
query heads `4j` to `4j + 3`, a contiguous block. Llama-2-7B, Code Llama 7B and the pinned
checkpoint have one key/value head per query head, so the same code edits one head there.

```python
attn = model.layers[5].self_attn
groups = model.num_heads // model.num_kv_heads   # 4 on 8B and 3.2-1B
with model.trace(prompt):
    attn.attention_values[:, 1] = 0             # key/value head 1
    heads = attn.attention_head_outputs.save()  # only query heads 4 to 7 change
```

## Queries and keys are read after the rotary embedding

`attention_queries` and `attention_keys` are rotated by position; the projections before it are
the outputs of `q_proj` and `k_proj`. The two agree at position 0, where the rotation is the
identity, and nowhere else. A key or query moved to another position keeps the rotation of the
position it was read at. The cosines and sines are computed once per forward by the native
`model.model.rotary_emb` and passed to every block; Llama-3.1, 3.2 and 3.3 use `rope_type`
`llama3` (`rope_theta` 500000, rescaled frequencies), Llama-2 the plain rotary with
`rope_theta` 10000. Both sides can be read in one trace:

```python
attn = model.layers[5].self_attn
with model.trace(prompt):
    q_raw = attn.source.self_q_proj_0.output.save()   # before the rotary
    q = attn.attention_queries.save()                 # after it
```

## Position 0 is a sink with a very large norm

On Llama-3.2-1B the residual stream at position 0 has a norm near 420 after blocks 1 and 5,
against a mean of 2.4 and 4.5 at the other positions, and heads put on average 0.67 to 0.87 of their attention
on it (blocks 0, 1, 5, 8 and 15, on a test prompt). The Llama-3 tokenizer prepends its beginning-of-text token
(id `128000`); without it the first real token takes the role (norm 865 after block 5). A mean over
positions that includes position 0 is dominated by it: slice `[:, 1:]` before averaging
activations for steering vectors, mean ablation or probes.

```python
with model.trace(prompt):
    resid = model.layers[5].layer_output.save()

resid[0].float().norm(dim=-1)        # 422 at position 0, 4 to 5 elsewhere on 3.2-1B
```

## Target tokens differ by tokenizer

On Llama-3, `"Paris"` (`60704`) and `" Paris"` (`12366`) are two whole-word tokens, and
`get_first_tokens(["Paris"], model)` returns both. Llama-2's SentencePiece tokenizer prefixes
the word marker itself: `"Paris"` and `" Paris"` both encode to `▁Paris` (`3681`), and
`get_first_tokens(["Paris"], model)` returns `[2177, 3681]`, where `2177` is the mid-word
fragment `Par`. Decode the target ids on the checkpoint you run.

## The readout and the embeddings

`model.logits` is `model.lm_head.output`: no cap and no scale, so `project_on_vocab` is
`lm_head(norm(hidden))` and a logit lens at the last block equals `logits`. The final norm's
gain is `model.norm.weight` as stored. Llama-3.2-1B and 3B tie `lm_head` to `embed_tokens`
(one parameter), so an edit to `embed_tokens.weight` is an edit to the unembedding; 8B, 70B and
Llama-2-7B do not.

## What this family module covers

The module serves every checkpoint whose config says `model_type` `llama`: besides Meta's
Llama-2, Llama-3, 3.1, 3.2 and 3.3, that includes Code Llama, TinyLlama,
DeepSeek-R1-Distill-Llama-8B, SmolLM2 and Helium-1-2B. Their settings differ: SmolLM2-135M
ties its embeddings, has 9 query heads over 3 key/value heads, and its tokenizer adds no
beginning-of-sequence token; Code Llama 7B has `rope_theta` 1000000. Llama 4 is the
`llama4_text` family.

## Sparse autoencoders and transcoders

- **Llama Scope** (`OpenMOSS-Team/Llama-Scope`), on Llama-3.1-8B, every block. The `R` SAEs read
  the stream after the block, `model.layers[i].layer_output`; `A` reads
  `model.layers[i].self_attn.attention_output`; `M` reads `model.layers[i].mlp.mlp_output`. The
  `TC` transcoders read the normed stream, `model.layers[i].mlp.input`, and predict `mlp_output`.
  Inputs are scaled to a norm of `sqrt(hidden_size)` before encoding.
- **EleutherAI** `EleutherAI/sae-llama-3-8b-32x`, on Meta-Llama-3-8B: residual-stream SAEs whose
  hook points `layers.i` are the block outputs, `model.layers[i].layer_output`, and `embed_tokens`
  is `model.token_embeddings`.
- **circuit-tracer** transcoders on Llama-3.2-1B, per-layer (`mntss/transcoder-Llama-3.2-1B`)
  and cross-layer (`mntss/clt-llama-3.2-1b-524k`). They read `hook_resid_mid`, the stream before
  the MLP's norm, `model.layers[i].post_attention_layernorm.input`, and write `hook_mlp_out`,
  `model.layers[i].mlp.mlp_output`:

```python
with model.trace(prompt):
    mid = model.layers[i].post_attention_layernorm.input.save()   # hook_resid_mid
    out = model.layers[i].mlp.mlp_output.save()                    # hook_mlp_out
```
"""
