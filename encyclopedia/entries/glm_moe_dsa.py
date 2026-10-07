"""GLM-5: latent attention with DeepSeek Sparse Attention, the selection carried between blocks, and a sigmoid-routed mixture."""

MODEL_TYPE = "glm_moe_dsa"
TITLE = "GLM-5"
SUBTITLE = (
    "Latent attention behind an indexer that keeps each query's top keys, a selection the block returns "
    "beside the stream for the next block to reuse, and a sigmoid-routed mixture with a shared expert after three dense blocks."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "zai-org/GLM-5"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-GlmMoeDsaForCausalLM"
#: Every public id whose config is glm_moe_dsa. GLM-5.3-Flash is glm5_next, another family.
CHECKPOINTS = [
    "zai-org/GLM-5", "zai-org/GLM-5-FP8", "zai-org/GLM-5.1", "zai-org/GLM-5.1-FP8",
    "zai-org/GLM-5.2", "zai-org/GLM-5.2-FP8", "zai-org/GLM-5.3", "zai-org/GLM-5.3-BF16",
]

#: Set by hues.py (lineage: GLM).
PALETTE = {"hue": 128}
VLLM = False
QUIRKS = ["latent-attention", "sparse-attention", "tuple-blocks", "interleaved-rotary", "mixture-of-experts", "dense-first-blocks"]

#: Shapes, identities, read orders, the rotary order, the selection and the routing were run on the pinned tiny
#: checkpoint (block 0 a mixture of 8 experts, top 2, block 1 dense; queries, keys and values 128 wide;
#: index_topk 2048) and on the suite's copy with index_topk 2. Sizes, scales, the indexer layout, the routing
#: settings and the tokenizer are the Hub configs', meta builds and tokenizers; no real weights were run
#: (no checkpoint of the family is small enough).

#: The sublayers in forward order; a block draws the MLP its ``mlp`` is (dense on blocks 0-2, a mixture after).
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
            "detail": "sparse latent, {num_heads} heads, q·k {qk_head_dim}, v {v_head_dim}",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
        {
            "host": "mlp",
            "kind": "moe",
            "label": "MoE",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "interior": [
                "router_logits", "expert_weights", "expert_indices",
                "expert_outputs", "routed_output", "shared_expert_output",
            ],
            "detail": "{num_experts} experts × {moe_intermediate_size}, top {top_k}",
        },
    ],
    # GLM-5.2 and GLM-5.3 mark each block's indexer; on GLM-5 and GLM-5.1 every block is full, so no row is drawn
    "tick_facets": [
        {"key": "indexer_types", "label": "indexer",
         "values": {"full": "own indexer", "shared": "reuses the previous block's selection"}},
    ],
}

STRIP = {
    "embed": "A plain lookup: token_embeddings equals layers[0].input. The tokenizer prepends nothing.",
    "layers": "Each block returns (hidden_states, topk_indices); layer_output is the first element. The checkpoints' "
              "multi-token-prediction block (num_nextn_predict_layers) is not built: layers holds num_hidden_layers blocks.",
    "head": "lm_head has its own weight (tie_word_embeddings is false); logits is lm_head.output.",
}

NOTES = """
## The block, in order

```
h, sel = x + self_attn(input_layernorm(x), sel)   # sparse latent attention; sel is the key selection
out    = h + mlp(post_attention_layernorm(h))     # dense on blocks 0-2, a mixture after
return out, sel
```

Llama's pre-norm block and Llama's names. The block returns `(hidden_states, topk_indices)`, and
the attention `(attn_output, attn_weights, topk_indices)`: the key selection travels to the next
block beside the stream. `layer_output` and `attention_output` are the first elements, and the
identity is the plain sum on a dense block and on a mixture block:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

`config.mlp_layer_types` says which blocks are dense: blocks 0 to 2 on every GLM-5 checkpoint, a
mixture on the other 75. The configs carry one multi-token-prediction block
(`num_nextn_predict_layers` 1), stored after the last block; transformers does not build it, so
`model.layers` has 78 blocks.

## Latent attention: every head's key comes from one latent

`attention_queries` and `attention_keys` are `qk_head_dim` wide (`qk_nope_head_dim +
qk_rope_head_dim`, 192 + 64 = 256), `attention_values` and `attention_head_outputs` `head_dim`
(`v_head_dim`, also 256). The config's `head_dim` is set to `qk_rope_head_dim` (64), which no
served value has. Queries go through a compressed rank (`q_lora_rank` 2048: `q_a_proj`,
`q_a_layernorm`, `q_b_proj`; `self_attn.q_proj` is `None`). Keys and values are expanded from one
latent per token (`kv_lora_rank` 512) for every head, so `attention_keys` has `num_heads` (64)
heads, and the last 64 dimensions of every head's key are one shared rotary key. The scores are
scaled by `qk_head_dim ** -0.5` (`self_attn.scaling`, 0.0625).

```python
attn = model.layers[1].self_attn
with model.trace(prompt):
    q = attn.attention_queries.save()          # [1, 64, seq, 256]
    k = attn.attention_keys.save()             # [1, 64, seq, 256]
    v = attn.attention_values.save()           # [1, 64, seq, 256]

assert torch.equal(k[:, 0, :, -64:], k[:, 1, :, -64:])   # one rotary key
```

## The rotary part is served de-interleaved

The attention rotates pair (2i, 2i + 1) of the projection into dimensions i and i + 32 of the
rotary part on every checkpoint (the forward calls the interleaved rotary whatever
`rope_interleave` says). `attention_queries[..., -64:]` and `attention_keys[..., -64:]` are in
that rotate-half order: at position 0, where rotary is the identity, the served rotary part is
`q_b_proj`'s even dimensions followed by its odd ones.

```python
with model.trace(prompt):
    proj = attn.q_b_proj.output.save()
with model.trace(prompt):
    q = attn.attention_queries.save()

rope = proj.view(1, -1, model.num_heads, model.qk_head_dim)[0, 0, 0, -64:]
torch.testing.assert_close(q[0, 0, 0, -64:], torch.cat([rope[0::2], rope[1::2]]))
```

## Sparse attention is dense below 2048 tokens

A lightning indexer (`self_attn.indexer`: 32 heads of 128, `index_n_heads` and `index_head_dim`)
scores every key for every query and keeps the `index_topk` (2048) best. The attention masks the
others to the dtype's minimum before the shared interface, under eager and under the default
`sdpa` load alike, so the default load runs the same sparse computation; the interior values need
`attn_implementation="eager"`. `attention_scores` carry that mask, so a dropped key reads like a
future one; `attention_probabilities` is the dense `[batch, heads, query, key]` pattern, exactly
zero outside each query's selection. While a prompt is no longer than 2048 tokens the selection
lists every key, and the pattern is plain causal: on a 7-token prompt row *t* has *t* + 1 nonzero
entries. The selection is `self_attn.output[2]`, `[batch, query, min(index_topk, seq)]` int32,
on every block; it has no standard name.

```python
with model.trace(prompt):
    probs = attn.attention_probabilities.save()
    sel = attn.output[2].save()                # [1, seq, min(2048, seq)]

rows = (probs[0, 0] != 0).sum(-1)
assert torch.equal(rows, torch.arange(1, rows.numel() + 1))   # plain causal
```

A written pattern is used as written, so a write can give weight to a key the indexer dropped.
The indexer runs under `torch.no_grad()`: no gradient reaches its projections, and the selection
does not move with a gradient-based attribution.

## The selection is shared between blocks on GLM-5.2 and GLM-5.3

`config.indexer_types` marks each block `"full"` (it has an `indexer` and selects its own keys)
or `"shared"` (no `indexer`; it reuses the selection of the block before). Every block is full on
GLM-5 and GLM-5.1. On GLM-5.2 and GLM-5.3, 21 blocks are full (0, 1, 2, then every fourth from 6)
and the 57 others shared: blocks 7 to 9 attend with block 6's selection.
`skip_layers` hands the block after a skipped range no selection, which a shared block refuses
with a `ValueError`: skip up to a full block or to the end.

## The mixture: sigmoid scores, a selection bias, renormalized weights

256 routed experts, 8 per token, and one shared expert every token runs through.
`router_logits` are the logits before the sigmoid, and before the selection bias
(`router.e_score_correction_bias`). The bias is added to the sigmoids only to choose; `n_group`
and `topk_group` are 1, so every expert is a candidate. `expert_weights` are the chosen experts'
sigmoids without the bias, renormalized to sum to one (`norm_topk_prob`) and multiplied by
`routed_scaling_factor`, so they sum to 2.5 at every token.

```python
moe = model.layers[3].mlp
with model.trace(prompt):
    logits = moe.router_logits.save()
    w = moe.expert_weights.save()
    idx = moe.expert_indices.save()
    routed = moe.routed_output.save()
    shared = moe.shared_expert_output.save()
    out = moe.mlp_output.save()

chosen = logits.sigmoid().gather(-1, idx)
scale = model.config.routed_scaling_factor
torch.testing.assert_close(w, chosen / chosen.sum(-1, keepdim=True) * scale)
torch.testing.assert_close(routed + shared, out)
```

The shared expert runs after the routed ones, so `shared_expert_output` is read after
`routed_output`. Ablating an expert is zeroing the weight of every slot that chose it; the other
slots keep their weights, so the token's weights then sum to less than 2.5:

```python
with model.trace(prompt):
    moe.expert_weights = moe.expert_weights.masked_fill(moe.expert_indices == 3, 0)
    ablated = model.logits.save()
```

Blocks 0 to 2 have a dense MLP: `support()` reports every mixture value missing there.
"""
