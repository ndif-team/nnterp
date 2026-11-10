"""DeepSeek-V4: four parallel residual streams mixed by hyper-connections, compressed multi-query attention with a sink, all-MoE."""

MODEL_TYPE = "deepseek_v4"
TITLE = "DeepSeek-V4"

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "deepseek-ai/DeepSeek-V4-Flash"
#: The tiny checkpoint the test suite builds the page from (7 blocks, 4 streams, hidden 8; the suite loads it in float32).
PINNED = "yujiepan/deepseek-v4-bf16-tiny-random"
#: Every public id whose config is deepseek_v4. DeepSeek-V4.1-Flash is deepseek_v41, another family.
#: -Flash-Vision-Exp carries vision keys that DeepseekV4Config ignores: it builds as the text model.
CHECKPOINTS = [
    "deepseek-ai/DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4-Flash-Base", "deepseek-ai/DeepSeek-V4-Flash-0731",
    "deepseek-ai/DeepSeek-V4-Flash-DSpark", "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
    "deepseek-ai/DeepSeek-V4-Pro", "deepseek-ai/DeepSeek-V4-Pro-Base", "deepseek-ai/DeepSeek-V4-Pro-0813",
    "deepseek-ai/DeepSeek-V4-Pro-DSpark",
]

#: Set by hues.py (lineage: DeepSeek).
PALETTE = {"hue": 331}
QUIRKS = [
    "hyper-connections", "attention-sink", "sliding-window", "sparse-attention", "interleaved-rotary",
    "mixture-of-experts", "hash-routing",
]

#: Shapes, the block's formula, the stream mean, the readout, the key axis of each block type, the selection and the
#: routing (both routers) were run on the pinned tiny checkpoint in float32 on CPU (7 blocks: 2 sliding, then
#: compressed-sparse and heavily-compressed alternating; 3 hash blocks; 4 streams; heads 128 wide). Sizes, block
#: types, scales and routing settings are the Hub configs', read on meta builds; no real weights were run (no
#: checkpoint of the family is small enough).

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
            "detail": "MQA, {num_heads} heads × {head_dim}, k = v",
            "variants": {
                "sliding_attention": "MQA × {num_heads}, window {sliding_window}",
                "compressed_sparse_attention": "MQA × {num_heads}, window + indexed",
                "heavily_compressed_attention": "MQA × {num_heads}, window + coarse",
            },
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
    "identity": "mlp_combᵀ · (attention_combᵀ · layers[i].input + attention_post ⊗ self_attn.attention_output) "
                "+ mlp_post ⊗ mlp.mlp_output == layer_output",
    "identity_note": "combᵀ · x mixes the streams (comb.mT @ x); post ⊗ y writes y into every stream, scaled "
                     "(post[..., None] * y[..., None, :]).",
    "streams": {
        "count_key": "hc_mult",
        "collapse": {"self_attn": "attention_comb", "mlp": "mlp_comb"},
        "post": {"self_attn": "attention_post", "mlp": "mlp_post"},
        "readout": "hc_head",
    },
}

STRIP = {
    "embed": "Copied into each stream: layers[0].input is token_embeddings expanded to [batch, seq, streams, hidden_size].",
    "readout": "hc_head sums the last block's streams, each weighted by its content; project_on_vocab applies it.",
}

NOTES = """
## The residual is four streams

```
post_a, comb_a, xa = attn_hc(x)                  # x: [batch, seq, 4, hidden_size]
h   = comb_aᵀ · x + post_a ⊗ self_attn(input_layernorm(xa))
post_f, comb_f, xf = ffn_hc(h)
out = comb_fᵀ · h + post_f ⊗ mlp(post_attention_layernorm(xf))
```

Code written for a rank-3 residual runs on `layer_output` without an error and gives one answer per
stream. Index one stream or take the stream mean. Averaged over the streams the block is a sum, each
contribution weighted by the mean of its `post`. A vector added to every stream reaches the next
sublayer scaled by the sum of that sublayer's collapse weights, so a steering coefficient tuned on
another family does not carry over.

## The lens goes through `hc_head`

The model sums the last block's streams with `model.hc_head`, each weighted by its content, before
`norm` and `lm_head`. `project_on_vocab` does the same on a four-stream tensor, so the lens on the
last block equals `logits`. Given one stream or a contribution it skips `hc_head`, and that lens is
not `logits`.

## The keys are the values

The attention passes one tensor as both its keys and its values, so an in-place edit of
`attention_keys` edits `attention_values` too. Assign a new tensor to edit one alone.

## Compressed blocks have more keys than tokens

On a `compressed_sparse_attention` or `heavily_compressed_attention` block, compressed entries
follow the token keys, so the key axis of `attention_keys`, `attention_scores` and
`attention_probabilities` is `seq + seq // rate` long.

## A patch before block 2 does not move the hash routing

Blocks 0 to 2 pick experts from a table of the token ids, so a patch of `layer_output` from another
run at block 0 or 1 leaves the hash blocks after it routing by this run's ids. On every block
`expert_weights` sum to `routed_scaling_factor`.
"""
