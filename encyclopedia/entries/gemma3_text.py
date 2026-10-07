"""Gemma 3, text: Gemma 2's sandwich block with per-head query and key norms, and no softcaps."""

MODEL_TYPE = "gemma3_text"
TITLE = "Gemma 3"
SUBTITLE = (
    "Llama's tree with a sandwich block, so what reaches the residual stream is a post-norm's output; "
    "queries and keys are normed per head before the rotary, and five sliding-window blocks precede each full one."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "google/gemma-3-1b-pt"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-Gemma3ForCausalLM"
CHECKPOINTS = [
    "google/gemma-3-270m", "google/gemma-3-270m-it",
    "google/gemma-3-1b-pt", "google/gemma-3-1b-it",
    # Vision-language wrappers around a Gemma 3 text model: model_type gemma3, a key of WRAPPERS.
    "google/gemma-3-4b-pt", "google/gemma-3-4b-it",
    "google/gemma-3-12b-pt", "google/gemma-3-12b-it",
    "google/gemma-3-27b-pt", "google/gemma-3-27b-it",
]

#: The vision-language wrapper of this family, keyed by the wrapper's config.model_type. The vision encoder comes
#: from the checkpoint's vision_config.model_type (encyclopedia/vision/); what a config does not say is here.
WRAPPERS = {
    "gemma3": {
        "title": "Gemma 3",
        "pinned": "yujiepan/gemma-3-tiny-random",
        "projector": "multi_modal_projector: a 4 × 4 average pool over vision.tower_output's 64 × 64 patches, "
                     "an RMSNorm (mm_soft_emb_norm) and a matrix product (mm_input_projection_weight)",
        "projector_input": "`vision.tower_output`",
        "quirks": ["pooled-projector"],
        "notes": """
## The projector pools `tower_output`

`model.multi_modal_projector` (the `projector`) receives `vision.tower_output`, the patches after
`post_layernorm`, so `model.projector.input == model.vision.tower_output` and a write to the last
block or to `tower_output` reaches the text model. It lays the 4096 patches out as their 64 × 64
grid, averages each 4 × 4 block (`avg_pool`), norms the 256 averages with `mm_soft_emb_norm`
(gain `1 + weight`) and multiplies them by `mm_input_projection_weight`, `[vision_hidden, hidden]`.
`vision.image_features` is `model.projector.output` flattened over the images, `[256, hidden_size]`
per image. On the pinned tiny checkpoint:

```python
with model.trace(prompt, images=[image]):
    mask = model.vision.image_token_mask.save()
    out = model.vision.tower_output.save()
    fed = model.projector.input.save()
    features = model.vision.image_features.save()
    first = model.layers[0].input.save()

torch.equal(fed, out)                    # True: the projector reads tower_output
features.shape[0], out.shape[1]          # 256, 4096
torch.equal(first[mask], features)       # True: the scatter
```

## Every image is 896 × 896 pixels, 256 image tokens

`vision.image_size` is 896 on every size, and the processor resizes each image to it:
`pixel_values` is `[images, 3, 896, 896]`, 64 × 64 patches of 14 pixels, and the projector turns
them into 256 image tokens (`mm_tokens_per_image`). Every image gives the same count, so the
features of one image can be assigned into another's run.

## Eager attention keeps a 4096 × 4096 pattern per head

The vision encoder attends over all 4096 patches, so under `attn_implementation="eager"` every block's
pattern is 16 × 4096 × 4096. A trace keeps them all for autograd unless it runs under
`torch.no_grad()`: on `google/gemma-3-4b-pt` an eager trace without it runs out of a 48 GB card, and
with it peaks at 11 GB.

```python
with torch.no_grad(), model.trace(prompt, images=[image]):
    pattern = model.vision.layers[0].self_attn.attention_probabilities.save()
```
""",
    },
}

PALETTE = {"hue": 157}
VLLM = True
QUIRKS = ["sandwich-norms", "qk-norm", "sliding-window", "scaled-embeddings", "gain-norm"]

#: What the visualization draws: the sublayers in forward order, each with its norms,
#: the standard value that is its contribution, and the values read inside it.
#: ``detail`` and ``variants`` are formatted with the sizes and the config keys.
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
            "detail": "query heads {num_heads}, key/value heads {num_kv_heads}, head_dim {head_dim}; "
                      "q_norm and k_norm per head before the rotary; scores scaled by {query_pre_attn_scalar}^-1/2",
            "variants": {
                "sliding_attention": "{sliding_window}-token window, local RoPE",
                "full_attention": "full causal, global RoPE",
            },
            "post_norm_note": "On Gemma 3 this norm follows the attention. On Llama the same name is the norm before the MLP.",
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
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "The embedding module multiplies its lookup by √hidden_size, cast to the weight's dtype (33.94 on 1B, "
             "34.0 in bfloat16), so token_embeddings is the scaled tensor and equals layers[0].input on text.",
    "norm": "Gemma3RMSNorm: the gain is 1 + norm.weight, and the stored weights are large (mean 8.9 on 1B's final norm).",
    "head": "lm_head shares its weight with embed_tokens (tie_word_embeddings). Nothing follows it: "
            "final_logit_softcapping is null on every released size, so logits equals lm_head.output.",
}

NOTES = """
## The block, in order

```
a    = input_layernorm(x)
q, k = rope(q_norm(q_proj(a))), rope(k_norm(k_proj(a)))    # per head
h    = x + post_attention_layernorm(o_proj(attend(q, k, v_proj(a))))
out  = h + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(h)))
```

Four RMSNorms on the block and two inside the attention. `post_attention_layernorm` is the norm
*after* the attention, not Llama's pre-MLP norm; the MLP's input norm is
`pre_feedforward_layernorm`. `q_norm` and `k_norm` act on each head's `head_dim` vector; the
values are not normed.

## The contributions are the post-norms' outputs

`attention_output` and `mlp_output` are what the block adds: the outputs of
`post_attention_layernorm` and `post_feedforward_layernorm`. `self_attn.output[0]` and
`mlp.output` are the tensors entering those norms. The identity holds exactly, in float32 and
in bfloat16:

```python
with model.trace(prompt):
    x = model.layers[5].input.save()
    attn = model.layers[5].self_attn.attention_output.save()
    mlp = model.layers[5].mlp.mlp_output.save()
    out = model.layers[5].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

The post-norm sets the scale of what the stream receives, so scaling the module's output does
almost nothing. On 1B, halving `mlp.output` at block 5 moves the logits by at most 0.3 in
bfloat16 (0.03 in float32); halving `mlp_output` moves them by 2.9:

```python
with model.trace(prompt):
    model.layers[5].mlp.mlp_output[:] *= 0.5   # halves what the block adds
    model.layers[5].mlp.output[:] *= 0.5       # nearly a no-op
```

It is not exactly zero: bfloat16 rounds, and the norm's `eps` matters where the module's output
is small (the first position on 1B, RMS 0.004 at block 5).
Ablations, steering and direct logit attribution of a sublayer go through `attention_output` and
`mlp_output`.

## Eager and sdpa compute the same attention

`attn_logit_softcapping` and `final_logit_softcapping` are null in every released config, and
`Gemma3Attention` passes no softcap to the attention function in any case. A default (`sdpa`)
load runs the model's own computation: on 1B in float32 its logits are within 2e-5 of an eager
load's, on a 10-token and on a 701-token prompt. `attn_implementation="eager"` is needed only
for the values read inside the attention function (the ones marked `⚠`).

## Queries and keys are read after their norms and the rotary

`attention_queries` and `attention_keys` are the tensors the attention function receives:
`rope(q_norm(q_proj(x)))` and `rope(k_norm(k_proj(x)))`, per head. The normed keys before the
rotary are `self_attn.k_norm.output` (`[batch, kv_heads, seq, head_dim]`), the queries
`q_norm.output` (`heads` wide); the norms' gain is `1 + weight`, as on every Gemma 3 norm.
`attention_values` is `v_proj`'s output reshaped to heads.

The query scale is `query_pre_attn_scalar ** -0.5`. That is `256 ** -0.5` on 270M, 1B, 4B and
12B, where it equals `head_dim ** -0.5`; on 27B it is `168 ** -0.5` with `head_dim` 128.
`attention_scores` are the scaled scores plus the mask, at the softmax's input.

## Grouped-query attention

1B and 270M have 4 query heads and one key/value head: `attention_keys` and `attention_values`
are `[batch, 1, seq, 256]`, served before `repeat_kv`, and an edit to them reaches all four query
heads. 4B, 12B and 27B have two query heads per key/value head (8/4, 16/8, 32/16).

## Five sliding blocks to one full block

`config.layer_types` is `full_attention` on every sixth block (5, 11, 17, 23 on 1B's 26) and
`sliding_attention` elsewhere. The window is 512 tokens on 270M and 1B, 1024 on 4B, 12B and
27B; a query in a sliding block attends to itself and the 511 tokens before it on 1B. On a prompt
no longer than the window both masks are the same causal mask. A block's kind is
`model.layers[i].self_attn._module.sliding_window` (the window, or `None` on a full block).

The two kinds also use different rotary embeddings, from `config.rope_parameters`: base 10,000
on sliding blocks and 1,000,000 on full blocks, with linear scaling by 8 on the full blocks of 4B,
12B and 27B. The model computes both cos/sin tables and hands each block the one for its kind,
so `attention_queries` and `attention_keys` carry a different rotation in the two kinds of block.

## The readout has no softcap

`model.logits` is `model.lm_head.output`, and `project_on_vocab` is the final norm and
`lm_head` only, so a logit lens at the last block equals `logits` exactly:

```python
with model.trace(prompt):
    resid = model.layers[12].layer_output[:, -1].save()

lens = model.project_on_vocab(resid)    # norm, then lm_head; no cap
```

The vocabulary is 262,144 tokens on 270M and 1B and 262,208 on 4B, 12B and 27B, so logits from
the two groups are not the same width.

## The norm gain is 1 + weight, and the weights are not small

`Gemma3RMSNorm` multiplies by `1 + weight`. The stored weights are far from zero: on 1B the
final norm's weights average 8.9 (maximum 50) and block 5's `post_feedforward_layernorm` reaches 246.
Folding a norm into a neighbouring matrix uses `1 + model.norm._module.weight`.

## Embeddings are scaled and tied

`embed_tokens` multiplies its lookup by √hidden_size, cast to the weight's dtype first: 33.94 on
1B in float32 and 34.0 in bfloat16 (50.5 on 4B in bfloat16). `token_embeddings` is the scaled
tensor and equals `layers[0].input` on a text prompt. The embedding weight is the unembedding
(`tie_word_embeddings`), so an edit to `embed_tokens.weight` edits `lm_head` too.

## 4B, 12B and 27B are multimodal wrappers

270M and 1B are `Gemma3ForCausalLM`. 4B, 12B and 27B are `gemma3` checkpoints: the
text-generation task builds `Gemma3ForConditionalGeneration`, and nnterp picks this family from
`config.text_config` and binds `embed_tokens`, `layers` and `norm` under
`model.language_model`, with `lm_head` at the root. A text prompt runs the same block as above.

With an image, pass the processor's output to the trace (`model.trace(dict(inputs))`). The
vision features replace the 256 image tokens' embeddings after `embed_tokens`, so at those
positions `token_embeddings` is the placeholder token's embedding and `layers[0].input` is the
image; at text positions the two are equal. With the processor's `token_type_ids`, image tokens
attend to each other in both directions, in sliding and full blocks alike; text stays causal.

## Sparse autoencoders: Gemma Scope 2

Gemma Scope 2 (`google/gemma-scope-2-<size>-<pt|it>`, every size from 270M to 27B) publishes
SAEs, transcoders, crosscoders and, for 270M and 1B, cross-layer transcoders. Each `config.json`
names its hook point as a transformers module path:

```python
with model.trace(prompt):
    heads = model.layers[13].self_attn.o_proj.input.save()  # attn_out
    mlp_in = model.layers[13].mlp.input.save()       # transcoder input
    mlp_out = model.layers[13].mlp.mlp_output.save() # mlp_out, target
    resid = model.layers[13].layer_output.save()     # resid_post
```

The `attn_out` SAEs read `o_proj`'s input, the heads' outputs concatenated (`num_heads ×
head_dim`, 1024 on 1B), not `attention_output`; that tensor is `attention_head_outputs` flattened
over its last two axes, and `o_proj.input` is readable under any attention implementation. The
`mlp_out` SAEs read `post_feedforward_layernorm`'s output, which is `mlp_output`; the
transcoders map `pre_feedforward_layernorm`'s output (`mlp.input`) to it. The 4B, 12B and 27B
configs name `model.layers.<i>` paths, which on the wrapper are `model.language_model.layers.<i>`:
`model.layers[i]` either way.
"""
