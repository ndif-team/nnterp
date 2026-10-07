"""GPT-OSS: Llama's pre-norm block with a learned attention sink on every head, sliding and full blocks
alternating, and a mixture of experts whose experts are clamped SwiGLUs with biases."""

MODEL_TYPE = "gpt_oss"
TITLE = "GPT-OSS"
SUBTITLE = (
    "Llama's pre-norm block with a learned sink logit per head inside the softmax, so a pattern row sums to "
    "less than one, sliding-window and full blocks in turn, and a mixture of experts on every block."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "openai/gpt-oss-20b"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "yujiepan/gpt-oss-tiny-random"
CHECKPOINTS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]

#: Set by hues.py (no kin, in a gap between lineages).
PALETTE = {"hue": 148}
VLLM = False
QUIRKS = ["attention-sink", "sliding-window", "mixture-of-experts", "qkv-bias"]

#: The real checkpoints are 21B and 117B parameters: nothing here ran on real weights. Every identity and
#: shape in the notes, and every snippet, ran on the pinned tiny checkpoint (2 blocks, 32 experts, top 4);
#: the sizes are the 20b and 120b configs'.

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
            "detail": "{num_heads} heads × {head_dim}, {num_kv_heads} kv, sink",
            "variants": {
                "sliding_attention": "{num_heads}×{head_dim}, sink, window {sliding_window}",
                "full_attention": "{num_heads}×{head_dim}, sink, full causal",
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
                "expert_outputs", "routed_output",
            ],
            "detail": "{num_experts} experts × {intermediate_size}, top {top_k}",
        },
    ],
}

STRIP = {
    "embed": "A plain lookup: token_embeddings equals layers[0].input. The tokenizer prepends nothing.",
    "head": "lm_head has its own weight: tie_word_embeddings is false on 20b and 120b.",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))       # a sink column in the softmax
out = h + mlp(post_attention_layernorm(h))    # 4 experts per token
```

Llama's pre-norm block and Llama's names. The block returns a tensor and adds each sublayer's
output to the stream as it is, so the identity is the plain sum. The MLP module returns
`(hidden_states, router_scores)`; `mlp_output` is the first element, and `mlp.output` the tuple.

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

## Every head has a sink, so pattern rows sum to less than one

Each attention module holds `sinks`, one learned logit per query head (`[num_heads]`, 64 on 20b and
120b), on sliding and full blocks alike. The sink's share of the softmax mixes no value:
`attention_probabilities` has `seq` key columns whose rows sum to less than one, and
`1 - row sum` is the sink's share.

`attention_scores` is read at the masked scores just before the sink column joins, so it has the
pattern's shape and `softmax(attention_scores)` is not the pattern. Appending the column rebuilds it:

```python
attn = model.layers[0].self_attn
with model.trace(prompt):
    scores = attn.attention_scores.save()       # masked, before the sink joins
    pattern = attn.attention_probabilities.save()

assert (pattern.float().sum(-1) < 1).all()      # the sink keeps the rest

sinks = attn._module.sinks.to(scores.dtype)
column = sinks.view(1, -1, 1, 1).expand(scores.shape[0], -1, scores.shape[2], 1)
combined = torch.cat([scores, column], dim=-1)
combined = combined - combined.max(dim=-1, keepdim=True).values
full = combined.softmax(-1)
torch.testing.assert_close(full[..., :-1], pattern)
sink_share = full[..., -1]                      # [batch, heads, query]
```

The softmax runs in the scores' dtype, not in float32. A head's output is the values mixed by
these rows, so a row's sum is the share of that head's attention that reaches any token at that
query, and the rest goes to the sink. An edit to `attention_scores` is
renormalized against the sink; an edit to `attention_probabilities` sets the mix directly, and
its rows need not sum to anything.

## Sliding-window and full blocks alternate

`layer_types` alternates `sliding_attention` and `full_attention`, starting with a sliding block:
the even blocks attend over the latest `sliding_window` (128) positions and the odd blocks over the
whole prefix. On a prompt longer than the window, a sliding block's pattern is zero beyond it,
so it carries the sink column's share and at most 128 key columns per row:

```python
with model.trace(long_prompt):                        # more than 128 tokens
    pattern = model.layers[0].self_attn.attention_probabilities.save()

assert ((pattern[0, :, -1] > 0).sum(-1) <= 128).all()   # block 0 slides
```

## Loading: the default load is eager

transformers offers no SDPA path for GPT-OSS (`_supports_sdpa = False`), so a load without
`attn_implementation` runs the eager forward and every attention value is available. The 20b and 120b
configs carry an MXFP4 `quantization_config` for the experts only; attention, router, embeddings
and `lm_head` are not quantized.

## Attention: grouped heads, biased projections, YaRN rotary

64 query heads of width 64 share 8 key/value heads, so `attention_keys` and `attention_values` are
`[batch, 8, seq, 64]` and query head `h` reads key/value head `h // 8`: an edit to one key/value
head reaches 8 query heads. `q_proj`, `k_proj`, `v_proj` and `o_proj` all add a bias
(`attention_bias`). The rotary embedding is YaRN (`rope_scaling`: `factor` 32 over an original
4,096 positions) with `rope_theta` 150,000; `attention_queries` and `attention_keys` are read after it.

## The router takes a softmax over its top 4 logits

The router is a biased linear map to `num_experts` logits (32 on 20b, 128 on 120b); it keeps the 4
largest and takes a softmax over those 4 alone, so `expert_weights` sum to one at every token and
`router_logits` outside the top 4 do not enter the weights. There is no shared expert:
`routed_output` is `mlp_output`, and `expert_outputs` sums to it.

```python
moe = model.layers[1].mlp
with model.trace(prompt):
    logits = moe.router_logits.save()             # [batch, seq, num_experts]
    w = moe.expert_weights.save()                 # [batch, seq, top_k]
    idx = moe.expert_indices.save()
    each = moe.expert_outputs.save()              # [batch, seq, top_k, hidden]
    routed = moe.routed_output.save()

top = logits.topk(moe.top_k, dim=-1)
torch.testing.assert_close(top.values.softmax(-1).to(w.dtype), w)
assert torch.equal(top.indices, idx)
torch.testing.assert_close(each.sum(2), routed)
```

## The experts are clamped SwiGLUs with biases

The experts hold their weights as stacked parameters, `experts.gate_up_proj`
(`[num_experts, hidden, 2 × intermediate]`) and `experts.down_proj`, each with a bias. The
gate and up halves are interleaved, gate on the even columns and up on the odd ones. The gate is
clamped above at 7 and the up half to [−7, 7] (`swiglu_limit`), and the activation is
`(up + 1) · gate · sigmoid(1.702 · gate)`. An expert's output carries its `down_proj` bias, so
every slot adds a weighted bias, and `expert_outputs` is that output times the slot's weight.

## Ablating an expert touches only the tokens routed to it

Zeroing expert `e`'s weight where `expert_indices == e` removes that slot's term and leaves the
token's other 3 weights as they were (they are not renormalized). `mlp_output` changes on exactly
the tokens that chose `e`, and an expert no token chose has no effect:

```python
moe, e = model.layers[1].mlp, 3
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
The tokenizer prepends nothing, so `model.input_ids` is the prompt's tokens alone;
`<|startoftext|>` is id 199998 and the end-of-sequence token is `<|return|>`, id 200002.
"""
