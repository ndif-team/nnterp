"""DeepSeek-V3.2: V3's latent attention behind DeepSeek Sparse Attention, three dense first blocks, and a sigmoid-routed mixture."""

MODEL_TYPE = "deepseek_v32"
TITLE = "DeepSeek-V3.2"
SUBTITLE = (
    "Latent attention over only the 2048 keys a lightning indexer scores best for each query, and a mixture with a "
    "shared expert that chooses experts by sigmoid score plus a selection bias and weights them without the bias."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "deepseek-ai/DeepSeek-V3.2"
#: The tiny checkpoint the test suite builds the page from (block 0 dense, block 1 a mixture; index_topk 2048).
PINNED = "hf-tiny-v2/tiny-random-DeepseekV32ForCausalLM"
#: Every public id whose config is deepseek_v32; the four share one set of sizes.
CHECKPOINTS = [
    "deepseek-ai/DeepSeek-V3.2", "deepseek-ai/DeepSeek-V3.2-Speciale",
    "deepseek-ai/DeepSeek-V3.2-Exp", "deepseek-ai/DeepSeek-V3.2-Exp-Base",
]

#: Set by hues.py (lineage: DeepSeek).
PALETTE = {"hue": 328}
QUIRKS = ["latent-attention", "sparse-attention", "interleaved-rotary", "mixture-of-experts", "dense-first-blocks"]

#: Shapes, identities, read orders, the rotary order, the selection and the routing were run on the pinned tiny
#: checkpoint (block 0 dense, block 1 a mixture of 8 experts, top 2; queries, keys and values 128 wide;
#: index_topk 2048) on CPU. Sizes, scales, the indexer layout, the routing settings and the tokenizer are the Hub
#: configs' and tokenizer's, read on meta builds; no real weights were run (no checkpoint of the family is small enough).

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
}

STRIP = {
    "embed": "A plain lookup: token_embeddings equals layers[0].input. The tokenizer prepends nothing.",
    "layers": "The checkpoints' multi-token-prediction block (num_nextn_predict_layers) is not built: layers holds num_hidden_layers blocks.",
    "head": "lm_head has its own weight (tie_word_embeddings is false); logits is lm_head.output.",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))      # sparse latent attention
out = h + mlp(post_attention_layernorm(h))   # dense on blocks 0-2, a mixture after
```

Llama's pre-norm block and Llama's names. The block adds each sublayer's output to the stream and
returns a tensor, so the identity is the plain sum on a dense block and on a mixture block:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

`config.mlp_layer_types` says which blocks are dense: blocks 0 to 2 on every checkpoint
(`first_k_dense_replace` 3), a mixture on the other 58. The configs carry one
multi-token-prediction block (`num_nextn_predict_layers` 1), stored after the last block;
transformers does not build it, so `model.layers` has 61 blocks.

## Latent attention: every head's key comes from one latent

`attention_queries` and `attention_keys` are `qk_head_dim` wide (`qk_nope_head_dim +
qk_rope_head_dim`, 128 + 64 = 192), `attention_values` and `attention_head_outputs` `head_dim`
(`v_head_dim`, 128). transformers sets the config's `head_dim` to `qk_rope_head_dim`, the rotary
width (64); no served value is that wide. Queries go through a compressed rank (`q_lora_rank`
1536): `q_a_proj`, `q_a_layernorm`, then `q_b_proj`, with no `q_proj` module. Keys and values
are expanded for every head from one latent per token (`kv_lora_rank` 512), so `attention_keys`
has `num_heads` (128) heads. The last 64 dimensions of every head's key are one rotary key that
all heads share.

```python
attn = model.layers[1].self_attn
with model.trace(prompt):
    q = attn.attention_queries.save()          # [1, 128, seq, 192]
    k = attn.attention_keys.save()             # [1, 128, seq, 192]
    v = attn.attention_values.save()           # [1, 128, seq, 128]

assert torch.equal(k[:, 0, :, -64:], k[:, 1, :, -64:])   # one rotary key
```

## The rotary part is served de-interleaved

The attention rotates pair (2i, 2i + 1) of the projection into dimensions i and i + 32 of the
rotary part. The forward always does this; the config has no `rope_interleave` key to switch it.
`attention_queries[..., -64:]` and `attention_keys[..., -64:]` are in that rotate-half order: at
position 0, where rotary is the identity, the served rotary part is `q_b_proj`'s even dimensions
followed by its odd ones.

```python
with model.trace(prompt):
    proj = attn.q_b_proj.output.save()
with model.trace(prompt):
    q = attn.attention_queries.save()

rope = proj.view(1, -1, model.num_heads, model.qk_head_dim)[0, 0, 0, -64:]
torch.testing.assert_close(q[0, 0, 0, -64:], torch.cat([rope[0::2], rope[1::2]]))
```

## The query scale carries YaRN's factor

Every checkpoint extends the context with YaRN (`rope_scaling` factor 40, `mscale_all_dim` 1.0),
and the attention multiplies the scores by `qk_head_dim ** -0.5` times 1.87
(`self_attn.scaling`, 0.1352), not by `qk_head_dim ** -0.5` (0.0722) alone; recompute
`attention_scores` from the queries and keys with `self_attn.scaling`.

## Up to 2048 tokens the sparse attention is plain causal attention

A lightning indexer (`self_attn.indexer`: `index_n_heads` 64 heads of `index_head_dim` 128)
scores every key for every query and keeps the `index_topk` (2048) best. The attention sets the
other keys' mask entries to the dtype's minimum before it calls the attention function. It does
this under eager and under the default `sdpa` load alike, so both loads run the same sparse
computation; the interior values need `attn_implementation="eager"`. `attention_scores` carry
that mask, so a dropped key reads like a future one. `attention_probabilities` is the dense
`[batch, num_heads, query, key]` pattern, exactly zero outside each query's selection.

The selection is `self_attn.indexer.output`, int32 key indices of shape
`[batch, query, min(index_topk, seq)]`; it has no standard name. The indexer ranks future keys
last but does not drop them. On a prompt of up to 2048 tokens each row therefore lists every
position, the causal mask removes the future keys, and the pattern is plain causal: row *t* has
*t* + 1 nonzero entries.

```python
with model.trace(prompt):
    sel = attn.indexer.output.save()           # [1, seq, min(2048, seq)]
    probs = attn.attention_probabilities.save()

rows = (probs[0, 0] != 0).sum(-1)
assert torch.equal(rows, torch.arange(1, rows.numel() + 1))   # plain causal
```

A written pattern is used as written, so a write can give weight to a key the indexer dropped.
The indexer runs under `torch.no_grad()`: no gradient reaches its projections, and the selection
does not move with a gradient-based attribution.

## The mixture: sigmoid scores, a selection bias, renormalized weights

256 routed experts, 8 per token, and one shared expert every token runs through.
`router_logits` are the float32 logits before the sigmoid, and before the selection bias
(`router.e_score_correction_bias`). The bias is added to the sigmoids only to choose: the experts
sit in `n_group` 8 groups, each group scored by the sum of its two best biased scores, and a token
takes its 8 experts from its best `topk_group` 4 groups. `expert_weights` are the chosen experts'
sigmoids without the bias, renormalized to sum to one (`norm_topk_prob`) and multiplied by
`routed_scaling_factor`, so they sum to 2.5 at every token. A written `router_logits` moves both
the choice and the weights; a written bias moves only the choice.

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
