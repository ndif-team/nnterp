"""Solar Open: Llama's block with grouped-query attention under YaRN, and DeepSeek-V3's mixture with a shared expert on every block."""

MODEL_TYPE = "solar_open"
TITLE = "Solar Open"
SUBTITLE = (
    "Llama's pre-norm block with 64 query heads over 8 key/value heads, and on every block a mixture of 128 "
    "experts, 8 per token from a sigmoid router whose correction bias steers the choice only, beside a shared "
    "expert every token runs."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "upstage/Solar-Open-100B"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "onnx-internal-testing/tiny-random-SolarOpenForCausalLM"
#: Upstage's one release with this config; the quantized copies on the Hub are other publishers'.
CHECKPOINTS = ["upstage/Solar-Open-100B"]

#: No kin among the entries; the free gap between persimmon (171) and cohere (176).
PALETTE = {"hue": 173}
VLLM = False
QUIRKS = ["mixture-of-experts"]

#: Every number in the notes is Solar-Open-100B's config or a value transformers computes from it; no real
#: weights were run (about 100B parameters). The snippets ran on the pinned tiny checkpoint (4 experts, top 2),
#: with its e_score_correction_bias moved off zero for the router check.

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
            "detail": "{num_heads} heads over {num_kv_heads} kv, head_dim {head_dim}",
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
    "embed": "A plain lookup: token_embeddings equals layers[0].input. The tokenizer prepends no BOS.",
    "head": "lm_head has its own weight: tie_word_embeddings is false.",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))
out = h + mlp(post_attention_layernorm(h))    # 8 of 128 experts, plus the shared expert
```

Llama's pre-norm block and Llama's names. Nothing norms or scales a sublayer's output, so
`attention_output` is `o_proj`'s output and `mlp_output` the mixture's, shared expert included, and
the identity is the plain sum, exact in float32 on the pinned tiny checkpoint:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

## Every block is a mixture

Transformers builds `SolarOpenMoE` on every block; the config's `first_k_dense_replace` is 0, and
`SolarOpenMLP` is only the shared expert's class. The experts are `moe_intermediate_size` (1280) wide,
and so is the one shared expert (`n_shared_experts` × 1280). `model.intermediate_size` reads the
config's `intermediate_size`, 10240, which no module of Solar-Open-100B has.

## The router: sigmoid scores, a bias for the choice, weights renormalized

`router` (native `gate`) scores the 128 experts in float32 and takes their sigmoid. It adds
`e_score_correction_bias` (a buffer) to choose the 8 experts and then reads the weights from the
unbiased sigmoid, renormalizes them to sum to one and multiplies them by `routed_scaling_factor`, 1.0
on Solar-Open-100B. `n_group` and `topk_group` are 1, so no group limits the choice. `router_logits`
is before the sigmoid:

```python
moe = model.layers[1].mlp
with model.trace(prompt):
    logits = moe.router_logits.save()       # [batch, seq, num_experts]
    w = moe.expert_weights.save()           # [batch, seq, top_k]
    idx = moe.expert_indices.save()

bias = moe._module.gate.e_score_correction_bias
scores = logits.float().sigmoid()
chosen = (scores + bias).topk(moe.top_k, dim=-1).indices
assert torch.equal(chosen.sort(-1).values, idx.sort(-1).values)   # biased choice
weights = scores.gather(-1, idx)
torch.testing.assert_close((weights / weights.sum(-1, keepdim=True)).to(w.dtype), w)
```

The slots are not sorted by weight. A logit written at `router_logits` changes both the choice and
the weights.

## The shared expert runs on every token

`mlp_output` is `routed_output + shared_expert_output`. The shared expert (`mlp.shared_experts`)
reads the same normed input as the router, and its output is added after the routed experts', so
read `shared_expert_output` after `routed_output` in one trace. An expert ablation leaves it alone:
zeroing expert `e`'s weight where `expert_indices == e` changes `mlp_output` on exactly the tokens
that chose `e`, and the remaining 7 weights are not renormalized, so those tokens' routed output
shrinks.

```python
moe, e = model.layers[1].mlp, 3
with model.trace(prompt):
    idx = moe.expert_indices.save()
    clean = moe.mlp_output.save()
with model.trace(prompt):
    moe.expert_weights = moe.expert_weights.masked_fill(moe.expert_indices == e, 0)
    ablated = moe.mlp_output.save()

assert torch.equal((ablated != clean).any(-1), (idx == e).any(-1))
```

## YaRN scales the queries and keys

Rotary covers the whole head (`partial_rotary_factor` 1.0) with YaRN, `factor` 2.0 over
`original_max_position_embeddings` 65536. YaRN multiplies the rotary cosines and sines by
`0.1 * ln(2) + 1 = 1.0693`, so `attention_queries` and `attention_keys` are 1.0693 times the rotated
projections, and the scores carry that factor twice (1.143), on top of `1 / sqrt(head_dim)`. Each of
the 64 query heads is 128 wide, so `q_proj` maps the 4096-wide stream to 8192; 8 query heads share
each key/value head, and an edit to one key or value head reaches all 8 of its query heads. The
attention interior needs `attn_implementation="eager"`.

## The readout and the tokenizer

`logits` is `lm_head.output`, with no cap or scale; `lm_head` and `embed_tokens` are separate
weights. The tokenizer prepends nothing, so `model.input_ids` is the prompt's tokens alone;
`<|endoftext|>` (id 2) is the end of text.
"""
