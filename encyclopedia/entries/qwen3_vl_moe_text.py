"""Qwen3-VL-MoE: the text model of the Qwen3-VL mixture-of-experts checkpoints, with DeepStack after its first blocks."""

MODEL_TYPE = "qwen3_vl_moe_text"
TITLE = "Qwen3-VL-MoE"
SUBTITLE = (
    "Qwen3's attention, queries and keys RMS-normed per head, under a multimodal rotary embedding, and a "
    "mixture of 128 experts on every block; on an image prompt the text model adds vision features at the "
    "image positions again after blocks 0, 1 and 2, outside the blocks."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "Qwen/Qwen3-VL-30B-A3B-Instruct"
#: The tiny checkpoint the test suite builds the page from: the wrapper itself, as every checkpoint is.
PINNED = "yujiepan/qwen3-vl-moe-tiny-random"
#: Every Qwen3-VL-MoE checkpoint is the vision-language wrapper (model_type qwen3_vl_moe); there is no text-only release.
CHECKPOINTS = [
    "Qwen/Qwen3-VL-30B-A3B-Instruct", "Qwen/Qwen3-VL-30B-A3B-Thinking",
    "Qwen/Qwen3-VL-235B-A22B-Instruct", "Qwen/Qwen3-VL-235B-A22B-Thinking",
]

#: The vision-language wrapper every checkpoint above is, keyed by its config.model_type. The vision encoder comes from
#: the checkpoint's vision_config.model_type (encyclopedia/vision/qwen_vit.py); what a config does not say is here.
#: Shapes and identities were checked on the pinned tiny wrapper (and on a copy with five text blocks, for the blocks
#: past the taps); the numbers are from docs/usage/vision.md, on the dense Qwen3-VL-4B-Instruct.
WRAPPERS = {
    "qwen3_vl_moe": {
        "title": "Qwen3-VL-MoE",
        "pinned": "yujiepan/qwen3-vl-moe-tiny-random",
        "projector": "merger, inside the vision encoder: a LayerNorm on each patch, then an MLP (linear_fc1, GELU, "
                     "linear_fc2) over each 2 × 2 block of patches concatenated, one image token per block",
        "projector_input": "the merger's input: the last block's output",
        "quirks": ["deepstack"],
        "notes": """
## The merger folds four patches into one image token

`model.projector` is `model.visual.merger`, the vision encoder's last module: `norm` (a LayerNorm over
`vision_hidden`) on each patch, then each 2 × 2 block of consecutive patches concatenated into one
`4 * vision_hidden` vector, `linear_fc1`, GELU and `linear_fc2` to the text model's width. An image of
`t * h * w` patches (its `image_grid_thw` row) is `t * h * w / 4` image tokens, and
`vision.image_features` is `model.projector.output` as it is, `[image_tokens, hidden]`. A patch is 16
pixels, so an image token covers 32 × 32. The merger maps 4 × 1152 = 4608 to 2048 on 30B-A3B and to
4096 on 235B-A22B.

```python
with model.trace(prompt, images=[red, wide]):
    mask = model.vision.image_token_mask.save()
    fed = model.projector.input.save()          # [patches, vision_hidden]
    features = model.vision.image_features.save()
    first = model.layers[0].input.save()

fed.shape[0] // 4 == mask.sum()                 # True: one image token per 2 x 2 block
torch.equal(first[mask], features)              # True
```

## A learned position embedding follows patch_embed

The vision encoder adds `pos_embed`, a learned table of 48 × 48 positions resampled to each image's
grid, to the patch embedding before block 0, so `vision.layers[0].input` is not
`vision.patch_embeddings`. Positions also enter every vision block through the 2D rotary embedding.

## DeepStack: three more mergers, under their native names

`vision_config.deepstack_visual_indexes` names three vision blocks, 8, 16 and 24 of 27, on every
checkpoint. After each, its own merger, `vision.deepstack_merger_list[k]`, reads that block's output and
folds it to image tokens as the main merger does, with its LayerNorm after the 2 × 2 concatenation
(over `4 * vision_hidden`) instead of before it. The mergers keep their native path; their outputs reach
the text model as `model.layers[k].deepstack_output`, added at the image positions after text block `k`:

```python
with model.trace(prompt, images=[red]):
    tapped = model.vision.deepstack_merger_list[0].output.save()
    added = model.layers[0].deepstack_output.save()

torch.equal(added, tapped)                       # True
```

So `vision.image_features` is not the only way the image reaches the text model. On the dense
`Qwen/Qwen3-VL-4B-Instruct`, which has the same DeepStack, asked the colour of a red square on white,
zeroing `vision.image_features` alone still answers "Red"; zeroing `deepstack_output` on blocks 0 to 2
as well answers "White".

```python
with model.trace(prompt, images=[red]):
    model.vision.image_features[:] = 0
    for k in range(3):
        model.layers[k].deepstack_output[:] = 0
    ablated = model.logits.save()
```

## Read the vision values in the forward's order

The deepstack mergers run inside the vision encoder, each right after the block it taps, and the
merger runs inside it too, before the encoder returns. So in one trace read a tapped block's
`layer_output`, then `deepstack_merger_list[k]`, before any later vision block; `model.projector.input`
and `.output` before `vision.tower_output`; then `vision.image_features`; then, per text block,
`layer_output`, `deepstack_output`, and the next block's `input`. A read out of this order raises
`OutOfOrderError`.
""",
    },
}

#: Set by hues.py (lineage: Qwen).
PALETTE = {"hue": 304}
VLLM = False
QUIRKS = ["qk-norm", "multimodal-rotary", "mixture-of-experts"]

#: What the visualization draws. The DeepStack add after blocks 0-2 is the text model's, outside the block: `between`
#: draws it after `layer_output` on those blocks. Every block of every checkpoint, and of the pinned tiny, is a mixture.
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
            "detail": "{num_heads} heads, {num_kv_heads} kv, M-RoPE",
        },
        {
            "host": "mlp",
            "kind": "moe",
            "label": "MoE",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "interior": [
                "router_logits", "expert_weights", "expert_indices",
                "expert_outputs", "routed_output",
            ],
            "detail": "{num_experts} experts × {moe_intermediate_size}, top {top_k}",
        },
    ],
    "identity_note": "The contribution identity nnterp's suite checks on this family. On an image prompt the text "
                     "model then adds layers[k].deepstack_output at the image positions after blocks 0, 1 and 2, "
                     "outside the block: layers[k+1].input[mask] == layers[k].layer_output[mask] + "
                     "layers[k].deepstack_output there, layers[k+1].input == layers[k].layer_output elsewhere.",
    "between": [
        {
            "value": "deepstack_output",
            "mask": "vision.image_token_mask",
            "blocks": [0, 1, 2],
            "label": "DeepStack",
            "note": "Its rows are `vision.deepstack_merger_list[k].output`, the k-th DeepStack merger's image tokens. "
                    "A text prompt makes no DeepStack call, and `deepstack_output` is not reached.",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "A plain lookup, unscaled: token_embeddings equals layers[0].input on a text prompt. No BOS is "
             "prepended, so position 0 holds the text's first token.",
    "layers": "Between the blocks, on an image prompt: after blocks 0, 1 and 2 the text model adds "
              "layers[k].deepstack_output at the image positions, so there layers[k+1].input is not "
              "layers[k].layer_output.",
    "head": "lm_head has its own weight: tie_word_embeddings is false on every checkpoint. logits is "
            "lm_head.output, with no softcap or scale.",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))        # q_norm, k_norm per head, then M-RoPE
out = h + mlp(post_attention_layernorm(h))     # 8 of 128 experts per token
x   = out + deepstack[k] at the image tokens   # the text model, after blocks 0, 1, 2 only
```

The block is Qwen3-MoE's, with Llama's names. It adds the attention's output and the mixture's
output to the stream and returns a tensor, so the contribution identity is the plain sum; the
mixture has no shared expert, and `mlp_output` is `routed_output`:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    routed = model.layers[1].mlp.routed_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(routed, mlp)
torch.testing.assert_close(x + attn + mlp, out)
```

The third line is DeepStack, done by the text model between the blocks, on an image prompt only.

## The mixture: 128 experts, 8 per token, weights renormalized

`layers[i].mlp` is a mixture on every block (`decoder_sparse_step` 1, `mlp_only_layers` empty). The
router, `gate` aliased `router`, gives one logit per expert; the scoring is a softmax, the 8 largest
probabilities are kept, and `norm_topk_prob` divides them by their sum, so each token's
`expert_weights` sum to one:

```python
with model.trace(prompt):
    moe = model.layers[1].mlp
    logits = moe.router_logits.save()
    weights = moe.expert_weights.save()

top = logits.softmax(-1).topk(model.layers[1].mlp.top_k, dim=-1)
torch.testing.assert_close(weights, top.values / top.values.sum(-1, keepdim=True))
```

The experts are one fused module, `experts`, each a SwiGLU `moe_intermediate_size` wide (768 on
30B-A3B, 1536 on 235B-A22B); the config's `intermediate_size` (6144, 12288) is a dense width no block
uses. `expert_outputs` is each routing slot's weighted output, `[batch, seq, 8, hidden]`, and sums to
`routed_output`. Because the weights are renormalized, zeroing one expert's weight leaves the others'
unchanged and the remaining slots sum to less than one:

```python
with model.trace(prompt):
    moe = model.layers[1].mlp
    moe.expert_weights = moe.expert_weights.masked_fill(moe.expert_indices == e, 0)
    ablated = model.logits.save()
```

## After blocks 0 to 2, layer_output is not what the next block reads

On an image prompt the text model adds `deepstack_output` at the image positions between the first
blocks (the family module's docstring gives the identity); `vision.image_token_mask` marks them:

```python
with model.trace(prompt, images=[red]):
    mask = model.vision.image_token_mask.save()
    out = model.layers[0].layer_output.save()
    added = model.layers[0].deepstack_output.save()
    entering = model.layers[1].input.save()

torch.equal(entering[mask], out[mask] + added)   # True
torch.equal(entering[~mask], out[~mask])         # True
```

A patch from block 0's `layer_output` into block 1's `input` drops `deepstack_output`; a
residual-stream SAE or probe at `layer_output` on blocks 0 to 2 sees the stream before the add.
Ablating the image takes `vision.image_features` and all three `deepstack_output` (the Vision notes
give the numbers). A text prompt makes no deepstack call, and `deepstack_output` is not reached.

## Queries and keys: normed per head, then a multimodal rotary

`q_norm` and `k_norm` are RMSNorms over one head's 128 dimensions, applied before the rotary;
`attention_queries` and `attention_keys` are read at the attention interface, after both. The rotary
is M-RoPE: each token has a time, a height and a width position, and `mrope_section` `[24, 20, 20]`
assigns the 64 frequencies to the three, interleaved (`mrope_interleaved`). The model folds them into
one `cos` and `sin` before block 0. On text tokens the three positions are equal and count up by one.
An image's tokens share one time position and add their row and column in the merged grid to the
position the image starts at, and the text after the image continues from the largest of them plus one,
so a token's rotary position is smaller than its index in `input_ids` after an image. 30B-A3B has 32
query heads over 4 key/value heads, 235B-A22B 64 over 4; the score scale is `128 ** -0.5`.

## Loading

Every checkpoint is `Qwen3VLMoeForConditionalGeneration` (`model_type` `qwen3_vl_moe`), whose
`text_config` is `qwen3_vl_moe_text`. With `task="image-text-to-text"`, as this page builds them, the
wrapper loads with its processor: the text model at `model.language_model`, the vision encoder at
`model.visual` as `model.vision`. Transformers has no text-only class for it: a dispatched
`task="text-generation"` load builds the same wrapper without a processor, where every vision value and
`deepstack_output` raise `Unavailable`. `attn_implementation="eager"` is needed for the six attention
interior values on the text blocks; the vision encoder's pattern is `Unavailable` under every load.
`expert_outputs` needs `experts_implementation` `"grouped_mm"` (the default) or `"batched_mm"`.

## The prompt

The tokenizer sets `bos_token` to `None` and prepends nothing. The processor's chat template puts an
image as `<|vision_start|><|image_pad|><|vision_end|>`, and the processor repeats `<|image_pad|>`
(151655) once per image token; `vision.image_token_mask` is `input_ids == 151655`. The Instruct
templates end the prompt at `<|im_start|>assistant\\n`; the Thinking templates add `<think>\\n`, so the
next token starts a reasoning trace, not the answer.

## What loads as this family

Qwen3-VL-30B-A3B and 235B-A22B, Instruct and Thinking: 48 and 94 blocks. `rope_theta` is 5·10⁶ and
`max_position_embeddings` 262144 on each. The dense Qwen3-VL checkpoints, 2B to 32B, are
`qwen3_vl_text`; the text-only Qwen3-MoE checkpoints are `qwen3_moe`.
"""
