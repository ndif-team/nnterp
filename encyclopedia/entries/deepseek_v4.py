"""DeepSeek-V4: four parallel residual streams mixed by hyper-connections, compressed multi-query attention with a sink, all-MoE."""

MODEL_TYPE = "deepseek_v4"
TITLE = "DeepSeek-V4"
SUBTITLE = (
    "Four parallel residual streams that each sublayer reads through a learned collapse and writes back through "
    "per-stream weights and a stream-mixing matrix, around multi-query attention over a sliding window and "
    "compressed keys, and a mixture on every block."
)

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
VLLM = False
QUIRKS = ["hyper-connections", "attention-sink", "sliding-window", "sparse-attention", "interleaved-rotary", "mixture-of-experts"]

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
            "pre_norm_note": "This norm reads attn_hc's collapse of the four streams (attn_hc.output[2], a weighted sum "
                             "over the stream axis), not a stream: layers[i].input is [batch, seq, streams, hidden].",
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
            "pre_norm_note": "This norm reads ffn_hc's collapse of the four streams (ffn_hc.output[2]), not a stream.",
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
    "identity_note": "layers[i].input and layer_output are [batch, seq, streams, hidden]. combᵀ · x mixes the streams "
                     "(comb.transpose(-1, -2) @ x); post ⊗ y writes y into every stream (post[..., None] * y[..., None, :]). "
                     "Exact in float32 on every block of the tiny checkpoint; the stream mean is the additive part.",
}

STRIP = {
    "embed": "A plain lookup, copied into each of the hc_mult (4) streams: layers[0].input is token_embeddings "
             "expanded to [batch, seq, streams, hidden]. The tokenizer prepends nothing.",
    "layers": "Each block takes and returns the four streams. The checkpoints' multi-token-prediction blocks "
              "(num_nextn_predict_layers) are not built: layers holds num_hidden_layers blocks.",
    "norm": "norm reads hc_head's output, a learned, content-dependent weighted sum over the streams of the last "
            "layer_output; project_on_vocab applies hc_head before it.",
    "head": "lm_head has its own weight (tie_word_embeddings is false); logits is lm_head.output.",
}

NOTES = """
## The block, in order

```
post_a, comb_a, xa = attn_hc(x)                  # x: [batch, seq, 4, hidden]
h   = comb_aᵀ · x + post_a ⊗ self_attn(input_layernorm(xa))
post_f, comb_f, xf = ffn_hc(h)
out = comb_fᵀ · h + post_f ⊗ mlp(post_attention_layernorm(xf))
```

The residual is `hc_mult` (4) streams. Before each sublayer a hyper-connection (`attn_hc`,
`ffn_hc`) reads all four and returns `(post, comb, collapsed)`: `collapsed` is the sum of the
streams weighted by sigmoids of their content, and the sublayer's pre-norm reads it. The
sublayer's `[batch, seq, hidden]` output goes into every stream, scaled per stream by `post`, and
the streams themselves are mixed by `comb`, consumed transposed. Llama's norm names hold:
`input_layernorm` before the attention, `post_attention_layernorm` before the mixture.

## `layer_output` is four streams

`layer_output` and `layers[i].input` are `[batch, seq, streams, hidden]` (the `Streams`
layout); `attention_output` and `mlp_output` stay `[batch, seq, hidden]`. Block 0 reads the
embedding copied into each stream; after it the streams differ. Rank-3 code runs here and answers
per stream: `resid[:, -1]` is the last position's four streams, `[batch, streams, hidden]`.
Index one stream, or take the stream mean, which is the additive part of the block:

```python
with model.trace(prompt):
    out = model.layers[1].layer_output.save()   # [1, seq, 4, hidden]

last = out[:, -1, -1]          # the last position, the last stream: [1, hidden]
mean = out[:, -1].mean(-2)     # the last position, the stream mean: [1, hidden]
```

A write to `layer_output` lands natively; `steer` adds a `[hidden]` vector to every stream, and
`skip_layers` passes the streams through.

## The identity is the block's formula

The block is not a sum. The weights that put each contribution into the streams are the
block's own values: `attention_post` and `mlp_post`, `[batch, seq, streams]` (`StreamWeights`,
each in (0, 2)), and `attention_comb` and `mlp_comb`, `[batch, seq, streams, streams]`
(`StreamMixing`), read off `attn_hc.output` and `ffn_hc.output`. All four are float32 and
writable. The formula holds exactly in float32 on every block of the tiny checkpoint:

```python
layer = model.layers[3]
with model.trace(prompt):
    x = layer.input.save()
    post_a, comb_a = layer.attention_post.save(), layer.attention_comb.save()
    attn = layer.self_attn.attention_output.save()
    post_f, comb_f = layer.mlp_post.save(), layer.mlp_comb.save()
    mlp = layer.mlp.mlp_output.save()
    out = layer.layer_output.save()

h = comb_a.mT @ x + post_a[..., None] * attn[..., None, :]
torch.testing.assert_close(comb_f.mT @ h + post_f[..., None] * mlp[..., None, :], out)
```

`comb` is a Sinkhorn projection onto the doubly stochastic matrices: its columns sum to one, and
its rows to one up to the projection's residual. The stream mean is therefore additive, each
contribution weighted by its `post`'s mean, to about 2e-6 of the stream's largest value on the tiny checkpoint:

```python
mean = x.mean(2) + post_a.mean(-1, keepdim=True) * attn + post_f.mean(-1, keepdim=True) * mlp
torch.testing.assert_close(mean, out.mean(2), rtol=1e-4, atol=1e-4)
```

Zeroing `attention_output` removes the attention's write from every stream but leaves the
mixing; zeroing `attention_post` does the same through the weights. A direct logit attribution
needs the later blocks' `comb` and `post` too.

## The readout collapses the streams

The model reads the last block's streams out through `model.hc_head`, a learned,
content-dependent weighted sum over the stream axis, then `norm` and `lm_head`. The family's
`project_on_vocab` does the same on a `[batch, seq, streams, hidden]` tensor (or one position's
`[streams, hidden]`), so the lens on the last block is `logits` exactly. A rank-3 tensor (a
contribution, one stream) goes through `norm` and `lm_head` alone, and one stream's lens is not
`logits`.

```python
with model.trace(prompt):
    resid = model.layers[-1].layer_output.save()
    logits = model.logits.save()

assert torch.equal(model.project_on_vocab(resid), logits)
```

## Multi-query attention whose keys are its values

One key/value head (`num_key_value_heads` 1) serves all 64 query heads (128 on Pro), each 512
wide. The attention projects one `kv` per token and passes the same tensor as keys and values, so
`attention_keys` and `attention_values` are one object: an in-place edit of either edits both,
and an assignment gives one a tensor of its own. Queries are normed per head without a weight
(`q_b_norm`) and the scores are scaled by `head_dim ** -0.5`. Rotary turns the last 64
dimensions of each head (`partial_rotary_factor` 0.125) in adjacent pairs, so the values carry
it too; the module turns the attention's output back before the grouped output projection
(`o_a_proj` over `o_groups` 8, then `o_b_proj`), and `attention_head_outputs` is that rotated-back
tensor. A learned per-head sink (`self_attn.sinks`) joins the softmax, so pattern rows sum to less
than one. The family supports no attention implementation but eager, so the default load serves
the interior values.

## Three attention blocks: a window, and compressed keys after it

Every block attends to the latest `sliding_window` (128) tokens. `config.layer_types` adds
compressed keys on most blocks: a `compressed_sparse_attention` block appends one entry per 4
tokens, of which a lightning indexer (`self_attn.compressor.indexer`) keeps each query's
`index_topk` best (512 on Flash, 1024 on Pro); a `heavily_compressed_attention` block appends one
entry per 128 tokens and keeps all of them. The entries follow the token keys, so the key axis of
`attention_keys`, `attention_scores` and `attention_probabilities` is `seq + seq // rate` long
there, and an entry is visible to a query once its window has closed. The indexer runs before the
keys are joined, so its selection is read first. On a 13-token prompt the tiny checkpoint's block 2
has 16 keys:

```python
attn = model.layers[2].self_attn                 # compressed_sparse_attention
with model.trace(prompt):
    sel = attn.compressor.indexer.output.save()  # [1, 13, min(index_topk, 3)], -1 where none is closed
    k = attn.attention_keys.save()               # [1, 1, 13 + 13 // 4, 128]
```

Flash has two `sliding_attention` blocks (0 and 1), then `compressed_sparse_attention` and
`heavily_compressed_attention` alternating (21 and 20); Pro starts with two heavily compressed
blocks and alternates after them (30 and 31). The selection has no standard name; within the
compressed entries the pattern is zero outside it.

## The mixture: square-root-softplus scores, hashed on the first three blocks

Every block's MLP is a mixture: 256 routed experts (384 on Pro), 6 per token, and one shared
expert, with the gate and up projections clamped at `swiglu_limit` (10). `router_logits` are the
logits before the scoring, `sqrt(softplus(logits))` (`scoring_func` `sqrtsoftplus`). On a `moe`
block the token takes the 6 best scores plus a selection bias (`router.e_score_correction_bias`),
with no expert groups; on the `hash_moe` blocks (0 to 2) the experts are a fixed table of the
token id (`router.tid2eid[input_ids]`), so a run needs `input_ids`, and a written
`router_logits` changes `expert_weights` but not `expert_indices`. On both, `expert_weights` are
the chosen experts' scores without the bias, renormalized to one and multiplied by
`routed_scaling_factor` (1.5 on Flash, 2.5 on Pro).

```python
import torch.nn.functional as F

moe = model.layers[4].mlp
with model.trace(prompt):
    logits = moe.router_logits.save()
    w = moe.expert_weights.save()
    idx = moe.expert_indices.save()
    routed = moe.routed_output.save()
    shared = moe.shared_expert_output.save()
    out = moe.mlp_output.save()

chosen = F.softplus(logits).sqrt().gather(-1, idx)
scale = model.config.routed_scaling_factor
torch.testing.assert_close(w, chosen / chosen.sum(-1, keepdim=True) * scale)
torch.testing.assert_close(routed + shared, out)
```

`moe.SCORING` is `"hash"` on a hash block and `"sqrtsoftplus"` on the others. Ablating an expert
is zeroing the weight of every slot that chose it; its effect on the block is read on the streams,
one stream or their mean:

```python
e = idx[0, -1, 0].item()                       # an expert the last token uses
with model.trace(prompt):
    moe.expert_weights = moe.expert_weights.masked_fill(moe.expert_indices == e, 0)
    ablated = model.layers[4].layer_output.save()    # [1, seq, 4, hidden]

ablated[:, -1, -1]          # the last stream at the last position
ablated[:, -1].mean(-2)     # the stream mean at the last position
```
"""
