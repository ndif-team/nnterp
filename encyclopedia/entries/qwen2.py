"""Qwen2 and Qwen2.5, and every other line that loads as ``Qwen2ForCausalLM``."""

MODEL_TYPE = "qwen2"
TITLE = "Qwen2 / Qwen2.5"
SUBTITLE = (
    "Llama's block with a bias on the query, key and value projections; Qwen's tokenizer "
    "prepends no BOS, so position 0 is the text's first token and becomes the attention sink."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "Qwen/Qwen2.5-0.5B"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "yujiepan/qwen2-tiny-random"
CHECKPOINTS = [
    "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct-1M",
    "Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B", "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B", "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-Coder-1.5B", "Qwen/Qwen2.5-Coder-7B", "Qwen/Qwen2.5-Coder-32B-Instruct",
    "Qwen/Qwen2.5-Math-1.5B", "Qwen/Qwen2.5-Math-7B",
    "Qwen/Qwen2-0.5B", "Qwen/Qwen2-0.5B-Instruct", "Qwen/Qwen2-1.5B", "Qwen/Qwen2-7B", "Qwen/Qwen2-72B",
    "Qwen/Qwen1.5-0.5B",
    "Qwen/QwQ-32B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    # Vision-language wrappers around a Qwen2 text model, each a key of WRAPPERS by its config's model_type.
    "llava-hf/llava-interleave-qwen-0.5b-hf",  # model_type llava
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf", "llava-hf/llava-onevision-qwen2-7b-ov-hf",
]

#: The vision-language wrappers of this family, keyed by the wrapper's config.model_type. The vision encoder comes from
#: the checkpoint's vision_config.model_type (encyclopedia/vision/); what a config does not say is here.
WRAPPERS = {
    "llava": {
        "title": "llava-interleave",
        # The family suite's TestLlavaInterleaveVision runs on this real checkpoint (where CUDA is); the meta build is cheap.
        "pinned": "llava-hf/llava-interleave-qwen-0.5b-hf",
        "projector": "a two-layer MLP (linear_1, GELU, linear_2) over the last block's patches, before post_layernorm",
        "projector_input": "the last block's `layer_output`, before `vision.norm`",
        "notes": """
## The projector reads the last block, before the final norm

`vision_feature_layer` is `-1` and `vision_feature_select_strategy` is `"full"`: the projector
receives the vision encoder's last hidden state whole, which is `vision.layers[-1].layer_output`, the
stream before `post_layernorm`. `vision.tower_output`, the norm's output, is computed and
discarded, so a write to it does not reach the text model; a write to the last block's
`layer_output` does. Checked on LLaVA-OneVision's pinned tiny checkpoint, whose wrapper selects the
features with the same code and the same two settings:

```python
with model.trace(prompt, images=[image]):
    stream = model.vision.layers[-1].layer_output.save()
    fed = model.projector.input.save()

torch.equal(fed, stream)   # True
```

## One token per patch

`multi_modal_projector` maps each patch to one token: `linear_1` from the vision encoder's 1152 to the
text model's 1024, GELU, `linear_2`. The processor resizes every image to 384 × 384, so an image is
27 × 27 = 729 image tokens whatever its shape, and `vision.image_features` is
`model.projector.output` flattened over the images, `[729, 1024]` per image. The scatter is
`LlavaModel`'s, as on Llava 1.5.
""",
    },
    "llava_onevision": {
        "title": "LLaVA-OneVision",
        "pinned": "hf-tiny-v2/tiny-random-LlavaOnevisionForConditionalGeneration",
        "projector": "a two-layer MLP (linear_1, GELU, linear_2) over each crop's last-block patches; the wrapper unpads its output and adds newline tokens",
        "projector_input": "each crop's last-block `layer_output`, before `vision.norm`",
        "quirks": ["tiled-images", "unpadded-features"],
        "notes": """
## Crops as rows, and features that are not the projector's output

The processor cuts an image into a base image and crops at a resolution from
`image_grid_pinpoints` (36 of them, 384 to 2304 pixels a side), each a row of the vision encoder's
batch. The projector reads `vision.layers[-1].layer_output` of every crop whole
(`vision_feature_layer` `-1`, strategy `"full"`), before `post_layernorm`, so
`vision.tower_output` does not reach the text model. The wrapper then unpads the crops' features
to the image's aspect ratio and appends `image_newline` after each row of patches, so
`vision.image_features` has another row count than `model.projector.output`: the features are read
at the scatter, and `layers[0].input[vision.image_token_mask] == vision.image_features` holds.

On `llava-onevision-qwen2-0.5b-ov-hf` a 384 × 384 image is the base image and one crop, 2 × 729
projector rows, and 1485 image tokens: the base image's 729, the crop's 729 and 27 newlines. An
image 768 wide and 384 high is the base and two crops, and 2214 image tokens. On the pinned tiny checkpoint
the projector returns 8 rows (the base image and one crop, 4 patches each) and
`vision.image_features` 10, 2 of them `image_newline`. Edit `vision.image_features` for what the
text model receives.
""",
    },
}

#: Set by hues.py (lineage: Qwen).
PALETTE = {"hue": 282}
VLLM = True
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
            "detail": "{num_heads} heads, {num_kv_heads} kv; q, k, v biased",
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
    "embed": "An unscaled lookup: token_embeddings equals layers[0].input. The tokenizer prepends no BOS, "
             "so position 0 is the text's first token.",
    "head": "lm_head shares its weight with embed_tokens on the Qwen2 and Qwen2.5 checkpoints up to 3B "
            "(tie_word_embeddings); from 7B up it has its own. logits is lm_head.output, with no softcap or scale.",
}

NOTES = """
## Queries, keys and values carry a bias

`q_proj`, `k_proj` and `v_proj` add a bias; `o_proj` and the three MLP projections do not. The
bias is added before the rotary embedding, so it is rotated with the rest of the query and key:

```
q, k, v = q_proj(x), k_proj(x), v_proj(x)          # each W·x + b
q, k    = rope(q), rope(k)                         # the bias turns with the position
out     = o_proj(softmax(q·kᵀ / √head_dim) · v)    # o_proj: no bias
```

The biases are large. On 0.5B the query bias has a larger norm than the input-dependent part
`W_q·x` (median over tokens) on all 24 blocks, and the key bias on 8 of them; block 0's are 235
and 367 against 18 and 7. A query–key analysis from `W_q` and `W_k` alone leaves out the larger
part of every block's query. Zeroing the attention's input does not silence it: the queries, keys and values fall
back to their biases, and `attention_output` is `o_proj` of the value bias at every position.

```python
with model.trace(prompt):
    model.layers[5].self_attn.input[:] = 0
    k = model.layers[5].self_attn.attention_keys.save()      # rope(k_proj.bias), not zero
    attn = model.layers[5].self_attn.attention_output.save() # not zero either
```

To remove the attention's effect, ablate `attention_output` itself.

## Grouped-query attention

0.5B has 14 query heads over 2 key/value heads, so `attention_keys` and `attention_values` are
`[batch, 2, seq, 64]` and an edit to key/value head `j` reaches query heads `7j` to `7j + 6`.
The other sizes: 12 over 2 (1.5B), 16 over 2 (3B), 28 over 4 (7B), 40 over 8 (14B, 32B), 64 over
8 (72B); `head_dim` is 64 on 0.5B and 128 on the rest. The score scale is `head_dim ** -0.5`.
Of the Qwen1.5 checkpoints, which load in this family, 0.5B and 7B have no grouping
(`num_key_value_heads` equals `num_attention_heads`) and 32B and 110B group 5 and 8 query heads.

## The sliding-window keys do nothing

Every checkpoint on this page has `sliding_window` and `max_window_layers` in its `config.json`,
and every one sets `use_sliding_window: false`. `Qwen2Config` then sets `sliding_window` to `None` and every entry
of `layer_types` to `full_attention`, so every block attends causally over the whole prompt and
`model.layers[i].self_attn._module.sliding_window` is `None`. With no window and no softcap, the
default `sdpa` load and the eager one compute the same model: on 0.5B in float32 their logits
agree to 6e-5. `attn_implementation="eager"` is needed only for the six attention interior values.

## Position 0 is the first token, and it becomes the sink

The tokenizers prepend nothing (`bos_token` is `None`; the config's `bos_token_id`, 151643, is
`<|endoftext|>`, which no tokenizer call adds). Position 0 holds the text's first token, and
whatever that token is, the model turns it into an attention sink. On 0.5B, position 0's
`layer_output` points the same way on every prompt from block 2 to block 20 (cosine 0.999),
almost entirely along coordinate 62; from block 3 to block 20 its norm is 1400 to 1750, against
under 65 at every other position, and block 21 dissolves it. From block 3 on, a query puts on
average 0.55 to 0.95 of its attention on key 0.

So from block 2 to block 20 the first word has no representation of its own: a probe, a patch
or a logit lens at position 0 reads the sink. A mean over positions (for mean ablation, a steering vector,
a normalisation) is dominated by position 0 unless it is dropped. Prepending `<|endoftext|>`
moves the sink onto that token and leaves the first word its own position; on 0.5B the
next-token prediction barely moves (`' Paris'` 0.797 without, 0.807 with).

```python
with model.trace("<|endoftext|>The Eiffel Tower is in the city of"):
    resid = model.layers[10].layer_output.save()

resid[0].norm(dim=-1)        # position 0 about 1400, the others under 25
```

With no BOS to skip, `tokenizer(" Paris").input_ids[0]` is the word's own token here, but the
DeepSeek-R1-Distill-Qwen tokenizers do prepend a BOS (`<｜begin▁of▁sentence｜>`), where index 0
is that. `get_first_tokens` tokenizes with `add_special_tokens=False` and is right on both:
on Qwen2.5 it returns `'Paris'` and `' Paris'`.

## The chat template writes a system turn

The base checkpoints ship the same ChatML template as the instruct ones (`<|im_start|>`,
`<|im_end|>`). Given no system message, it writes one: `You are a helpful assistant.` on the
Qwen2 and Qwen2.5 base models, Qwen2-Instruct and Qwen2.5-7B-Instruct-1M; `You are Qwen, created
by Alibaba Cloud. You are a helpful assistant.` on Qwen2.5-Instruct and Qwen2.5-Coder; none on
Qwen2.5-Math and QwQ-32B. A templated prompt can therefore open with a system turn you did not
write, and its position 0 is `<|im_start|>`, which takes the sink. The instruct configs end
generation on `<|im_end|>` (151645), the base configs on `<|endoftext|>` (151643). The
DeepSeek-R1-Distill-Qwen tokenizers carry DeepSeek's own template instead.

## The readout is the plain projection

`model.logits` equals `model.lm_head.output`, and `project_on_vocab` on the last block's
`layer_output` equals `logits` exactly. The embedding and the unembedding are one matrix on
Qwen2 0.5B and 1.5B and Qwen2.5 0.5B, 1.5B and 3B, so an edit to `embed_tokens.weight` there
edits `lm_head`; from 7B up they are separate. DeepSeek-R1-Distill-Qwen-1.5B is untied although
Qwen2.5-Math-1.5B is tied.

## What loads as this family

Every checkpoint whose config says `Qwen2ForCausalLM`: Qwen2 and Qwen2.5 base and instruct,
Qwen2.5-Coder and Qwen2.5-Math, Qwen2.5-7B-Instruct-1M, the dense Qwen1.5 models, QwQ-32B, and
the DeepSeek-R1-Distill-Qwen models. `rope_theta` is 10000 on Qwen2.5-Math and the 1.5B and 7B
distills, 10⁷ on the 1M model and 10⁶ on the rest. Qwen2-57B-A14B and Qwen1.5-MoE are
`qwen2_moe`.

## Sparse autoencoders

`andyrdt/saes-qwen2.5-7b-instruct` holds BatchTopK SAEs on Qwen2.5-7B-Instruct, trained on the
output of blocks 3, 7, 11, 15, 19, 23 and 27 (`resid_post_layer_<i>`, `io: out`), which is
`model.layers[i].layer_output`; four per block, `k` 32 to 256. SAELens lists them as
`qwen2.5-7b-instruct-andyrdt`. Their training skipped each sample's first 8 tokens and dropped
activations above 10× the batch's median norm, so the sink at position 0 is outside what they
were trained on.
"""
