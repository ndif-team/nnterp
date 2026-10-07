"""MiMo-V2-Flash: sliding-window blocks with a sink and twice the key/value heads among full blocks, queries and keys
wider than values, and a sigmoid-routed mixture after a dense first block."""

MODEL_TYPE = "mimo_v2_flash"
TITLE = "MiMo-V2-Flash"
SUBTITLE = (
    "Sliding-window blocks with a learned sink in the softmax and twice the key/value heads of the full blocks, "
    "queries and keys wider than values, and a mixture of 256 experts chosen by sigmoid scores plus a bias."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "XiaomiMiMo/MiMo-V2-Flash"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-MiMoV2FlashForCausalLM"
#: MiMo-V2.5 and V2.6 have model_type mimo_v2, another architecture, and are not listed.
CHECKPOINTS = ["XiaomiMiMo/MiMo-V2-Flash", "XiaomiMiMo/MiMo-V2-Flash-Base"]

#: Set by hues.py (lineage: MiniMax / MiMo).
PALETTE = {"hue": 349}
VLLM = False
QUIRKS = [
    "attention-sink", "per-block-sizes", "sliding-window", "partial-rotary", "mixture-of-experts",
    "dense-first-blocks",
]

#: The checkpoints have 309B parameters: nothing here ran on real weights. Every identity, shape and snippet ran on
#: the pinned tiny checkpoint (block 0 full and dense, block 1 sliding and a mixture of 4 experts, top 2; queries and
#: keys 8 wide, values 16; float32); the sizes, the block layout and the rotary widths are the Hub configs', read on
#: meta builds.

#: The sublayers in forward order; a block draws the MLP its ``mlp`` is (dense on block 0, a mixture after).
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
            "detail": "{num_heads} heads, q·k {qk_head_dim}, v {v_head_dim}",
            "variants": {
                "full_attention": "{num_heads}×{qk_head_dim}/{v_head_dim}, {num_kv_heads} kv, full",
                "sliding_attention": "{num_heads}×{qk_head_dim}/{v_head_dim}, sink, window {sliding_window}",
            },
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
                "expert_outputs", "routed_output",
            ],
            "detail": "{num_experts} experts × {moe_intermediate_size}, top {top_k}",
        },
    ],
}

STRIP = {
    "embed": "A plain lookup: token_embeddings equals layers[0].input. The tokenizer prepends nothing.",
    "layers": "The checkpoints' multi-token-prediction weights (model.mtp) are not built: layers holds num_hidden_layers blocks.",
    "head": "lm_head has its own weight (tie_word_embeddings is false); logits is lm_head.output.",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))       # sliding: a sink, 2x kv heads
out = h + mlp(post_attention_layernorm(h))    # dense on block 0, then 8 of 256
```

Llama's pre-norm block and Llama's names. Block 0 has a dense MLP and the other 47 a mixture
(`mlp_layer_types`). The block adds each sublayer's output to the stream and returns a tensor, so the
identity is the plain sum on both block kinds:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

The checkpoints store multi-token-prediction blocks under `model.mtp`; transformers does not build
them, so `model.layers` has 48 blocks.

## Sliding blocks have twice the key/value heads

`layer_types` makes blocks 0, 5, 11, 17, 23, 29, 35, 41 and 47 full and the other 39 sliding, with a
window of 128 positions. Every block has 64 query heads, but a sliding block has 8 key/value heads,
twice `num_key_value_heads`, where a full block has 4. The root's `num_kv_heads` is the config's 4,
the full blocks' count; each block's own is on its attention:

```python
model.num_kv_heads                       # 4
model.layers[1].self_attn.num_kv_heads   # 8: block 1 slides
model.layers[0].self_attn.num_kv_heads   # 4: block 0 is full
model.layers[1].self_attn.num_heads      # 64 on every block
```

`attention_keys` and `attention_values` are `[batch, 8, seq, ·]` on a sliding block and
`[batch, 4, seq, ·]` on a full one, so a key/value head serves 8 query heads on a sliding block and
16 on a full one. The pattern has 64 heads on every block.

## Queries and keys are 192 wide, values 128

`attention_queries` and `attention_keys` are `qk_head_dim` wide (the config's `head_dim`, 192);
`attention_values` and `attention_head_outputs` are `head_dim` wide (`v_head_dim`, 128). The scores
are scaled by `qk_head_dim ** -0.5` (`self_attn.scaling`). The values are multiplied by
`attention_value_scale` (0.707) before the interface, so `attention_values` is `v_proj`'s output
times 0.707:

```python
attn = model.layers[0].self_attn
with model.trace(prompt):
    proj = attn.v_proj.output.save()
with model.trace(prompt):
    v = attn.attention_values.save()            # [batch, kv_heads, seq, head_dim]

heads = proj.view(1, -1, attn.num_kv_heads, model.head_dim).transpose(1, 2)
torch.testing.assert_close(heads * model.config.attention_value_scale, v)
```

Rotary turns the first 64 of each query and key head's 192 dimensions (`partial_rotary_factor`
0.334), with base 5,000,000 on the full blocks and 10,000 on the sliding ones; the other 128 carry
no position.

## A sliding block's sink takes part of every row

Each sliding block's attention holds `sinks`, one learned logit per query head (64); a full block's
`sinks` is `None`. The sink joins the softmax as one extra key column that mixes no value, so a
sliding block's pattern rows sum to less than one, and `1 - row sum` is the sink's share; a full
block's rows sum to one. `attention_scores` is read at the masked scores, before the sink column joins
and the row max is subtracted, so it has the pattern's shape and `softmax(attention_scores)` is not
the pattern on a sliding block. Appending the column rebuilds it:

```python
attn = model.layers[1].self_attn                # a sliding block
with model.trace(prompt):
    scores = attn.attention_scores.save()       # masked, before the sink joins
    pattern = attn.attention_probabilities.save()

assert (pattern.sum(-1) < 1).all()              # the sink keeps the rest

sinks = attn._module.sinks.to(scores.dtype)
column = sinks.view(1, -1, 1, 1).expand(scores.shape[0], -1, scores.shape[2], 1)
combined = torch.cat([scores, column], dim=-1)
combined = combined - combined.max(dim=-1, keepdim=True).values
full = combined.softmax(-1)
torch.testing.assert_close(full[..., :-1], pattern)
sink_share = full[..., -1]                      # [batch, heads, query]
```

The softmax runs in the scores' dtype. An edit to `attention_scores` on a sliding block is
renormalized against the sink; an edit to `attention_probabilities` sets the mix directly.

## Loading: the default load is eager

transformers offers no SDPA path for MiMo-V2-Flash (`_supports_sdpa = False`), so a load without
`attn_implementation` runs the eager forward and every attention value is available. The configs
carry an FP8 `quantization_config` (128 × 128 weight blocks) that leaves every block's `o_proj`
unquantized.

## The router: sigmoid scores, a selection bias, renormalized weights

The router is DeepSeek-V3's with one group (`n_group` and `topk_group` 1), so it takes a plain top 8.
`router_logits` is the projection, computed in float32. The router adds its selection bias,
`mlp.router.e_score_correction_bias`, to the sigmoids to choose; `expert_weights` are the chosen
experts' sigmoids without the bias, renormalized to sum to one (`norm_topk_prob`) and multiplied by
`routed_scaling_factor`, which the configs leave null and transformers sets to 1.0. There is no shared
expert, so `routed_output` is `mlp_output`.

```python
moe = model.layers[1].mlp
with model.trace(prompt):
    logits = moe.router_logits.save()           # [batch, seq, num_experts], float32
    w = moe.expert_weights.save()               # [batch, seq, top_k]
    idx = moe.expert_indices.save()
    each = moe.expert_outputs.save()            # [batch, seq, top_k, hidden]
    out = moe.mlp_output.save()

chosen = logits.sigmoid().gather(-1, idx)
scale = model.config.routed_scaling_factor
torch.testing.assert_close(w, chosen / chosen.sum(-1, keepdim=True) * scale)
torch.testing.assert_close(each.sum(2), out)    # no shared expert
```

A written `router_logits` moves both the choice and the weights; a written bias moves only the
choice. Block 0 has a dense MLP: `support()` reports every mixture value missing there.

## Ablating an expert touches only the tokens routed to it

Zeroing expert `e`'s weight where `expert_indices == e` removes that slot's term and leaves the
token's other weights as they were, so they then sum to less than one. `mlp_output` changes on exactly
the tokens that chose `e`:

```python
moe, e = model.layers[1].mlp, 0
with model.trace(prompt):
    idx = moe.expert_indices.save()
    clean = moe.mlp_output.save()
with model.trace(prompt):
    moe.expert_weights = moe.expert_weights.masked_fill(moe.expert_indices == e, 0)
    ablated = moe.mlp_output.save()

changed = (ablated != clean).any(-1)                       # [batch, seq]
assert torch.equal(changed, (idx == e).any(-1))            # only the tokens that chose e
```

## The readout and the tokenizer

`logits` is `lm_head.output`, with no cap or scale, and `project_on_vocab` on the last block's
`layer_output` equals `logits`. The final norm is a plain RMSNorm whose gain is `norm.weight`.
The tokenizer prepends nothing, so `model.input_ids` is the prompt's tokens alone. The end token is
`<|im_end|>` (id 151645) on MiMo-V2-Flash and `<|endoftext|>` (id 151643) on the Base checkpoint.
"""
