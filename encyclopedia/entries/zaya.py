"""ZAYA1: residual scaling modules after every sublayer, convolutional queries and keys, and a one-expert mixture with a skip class."""

MODEL_TYPE = "zaya"
TITLE = "ZAYA1"
SUBTITLE = (
    "A pre-norm block in which a learned per-channel scale and shift is applied to both the sublayer's output "
    "and the stream at every add, with queries and keys mixed by a causal convolution, and a mixture that "
    "runs one expert per token, routed with a skip class and a router state carried from block to block."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "Zyphra/ZAYA1-8B"
#: The tiny checkpoint the test suite builds the page from (the suite runs a copy with qk_norm.temp set to one).
PINNED = "hf-tiny-v2/tiny-random-ZayaForCausalLM"
#: The -base, -reasoning-base and -legacy checkpoints keep the original layout's config (`rope_scaling: false`),
#: which transformers 5.17 does not parse; the page lists them greyed out with that reason.
CHECKPOINTS = [
    "Zyphra/ZAYA1-8B", "Zyphra/ZAYA1-8B-FP8-Experts", "Zyphra/ZAYA1-8B-MXFP4-Experts", "Zyphra/ZAYA1-74B-preview",
    "Zyphra/ZAYA1-base", "Zyphra/ZAYA1-reasoning-base", "Zyphra/ZAYA1-8B-legacy", "Zyphra/ZAYA1-74B-preview-legacy",
]

#: Set by hues.py (no kin, in a gap between lineages).
PALETTE = {"hue": 354}
VLLM = False
QUIRKS = [
    "tuple-blocks", "scaled-residual-adds", "fp32-residual", "embedding-multiplier", "partial-rotary",
    "sliding-window", "mixture-of-experts", "unnormalized-routing",
]

#: The real numbers in the notes are ZAYA1-8B's parameters, read from its safetensors (and ZAYA1-74B-preview's
#: balancing_biases and qk_norm.temp); no ZAYA1 checkpoint was run (the smallest has 8B parameters). Every snippet
#: ran on the pinned tiny checkpoint with its merges moved off their initial values (tests/families/test_zaya.py).

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
            "detail": "{num_heads} heads over {num_kv_heads} kv, conv q/k",
            "variants": {
                "hybrid": "full causal, conv q/k",
                "hybrid_sliding": "sliding window, conv q/k",
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
            "detail": "{num_experts} experts × {intermediate_size}, top {top_k}, + skip",
            "pre_norm_note": "The MoE's input norm. It reads the stream after post_attention_residual_scale has "
                             "merged the attention into it.",
        },
    ],
    "identity": "h = (layers[i].input + a.residual_bias) * a.residual_scale + self_attn.attention_output; "
                "(h + m.residual_bias) * m.residual_scale + mlp.mlp_output == layer_output",
    "identity_note": "a is layers[i].post_attention_residual_scale._module and m is layers[i].post_mlp_residual_scale._module, "
                     "the block's two ZayaResidualScaling merges; the contributions are already scaled by each merge's "
                     "hidden_states_scale. Exact in float32 on the pinned tiny checkpoint with its merges moved.",
}

STRIP = {
    "embed": "layers[0].input is (token_embeddings + input_hidden_states_bias) * input_hidden_states_scale, cast to "
             "float32: two [hidden_size] parameters of the model (scale 0.65 to 3.03 on ZAYA1-8B, mean 1.94). "
             "The tokenizer prepends <bos>.",
    "layers": "Each block rescales the stream at both of its adds, so a block's terms reach the last stream multiplied, "
              "channel by channel, by every later merge's residual_scale. The stream stays float32.",
    "norm": "The final norm casts the float32 stream to its weight's dtype first; its weight runs 0.036 to 5.0 on "
            "ZAYA1-8B (mean 2.9).",
    "head": "lm_head shares its weight with embed_tokens (tie_word_embeddings) on ZAYA1-8B and 74B-preview.",
}

NOTES = """
## The block, in order

```
h   = post_attention_residual_scale(self_attn(input_layernorm(x)), x)
out = post_mlp_residual_scale(mlp(post_attention_layernorm(h)), h)

merge(o, r) = (o + hidden_states_bias) * hidden_states_scale
            + (r + residual_bias) * residual_scale
```

Each merge is a `ZayaResidualScaling` module on the block with four `[hidden_size]` parameters. The
block returns `(hidden_states, prev_router_hidden_states)`; `layer_output` is the first element.

## The contributions are the merges' scaled terms

`attention_output` is `(self_attn.output[0] + hidden_states_bias) * hidden_states_scale` with the
first merge's parameters, and `mlp_output` the same over the mixture's output with the second's: a
computed copy, divided back on assignment. The merge also rescales the stream, so the identity carries
both merges' `residual_scale` and `residual_bias`. It is exact in float32 on the pinned tiny checkpoint
with its merges moved off their initial values:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

a = model.layers[1].post_attention_residual_scale._module
m = model.layers[1].post_mlp_residual_scale._module
h = (x + a.residual_bias) * a.residual_scale + attn
torch.testing.assert_close((h + m.residual_bias) * m.residual_scale + mlp, out)
```

A write to a contribution is divided back by `hidden_states_scale` over the whole tensor, so the
positions it does not touch come back changed by rounding; load in float32 for small edits. Zeroing
`attention_output` removes the term and leaves `(x + residual_bias) * residual_scale`, not `x`.

## The merges shrink the stream

On ZAYA1-8B `post_attention_residual_scale.residual_scale` averages 0.52 to 2.03 per block and
`post_mlp_residual_scale.residual_scale` 0.046 to 1.19, and the product of all 80 `residual_scale`
vectors has a median of 6.8e-6 over the channels (at most 0.034). A term added early therefore reaches
the last stream multiplied by that product, channel by channel; a direct attribution to the logits
has to carry it. The `hidden_states_scale` that sizes each contribution reaches 3.8 on the attention
merges and 13.2 on the mixture's. The biases are at most 1.15 in size and average zero.

## The stream is float32 whatever the load

The model casts the embedding to float32 after its shift and scale, and each merge adds a float32
stream, so `layers[i].input` and `layer_output` are float32 under a bfloat16 load, while
`token_embeddings`, `attention_output`, `mlp_output` and `logits` keep the load's dtype.
`input_layernorm`, `post_attention_layernorm` and the final norm cast the stream to their weight's
dtype before they norm it.

## Queries and keys are convolved, normed to √`head_dim` and the keys scaled

`qkv_proj` computes the queries and keys with `q_proj` and `k_proj`, mixes them over the current and
two previous tokens with two causal convolutions (`cca_time0` and `cca_time1` are 2 on ZAYA1-8B), and adds back a residual of the projections. Half of
each value head is the current token's (`v_proj_current`) and half the previous token's
(`v_proj_delayed`). `qk_norm` then L2-normalizes every query and key head to length √`head_dim` and
multiplies the keys by `temp`, one per key/value head; rotary turns the first half of each head
(`partial_rotary_factor` 0.5). `attention_queries` and `attention_keys` are read after all of it, at
the attention interface. `temp` is 0.8 to 12.9 on ZAYA1-8B; on the pinned tiny checkpoint it is zero,
which zeroes every key, so the suite's copy sets it to one. ZAYA1-74B-preview attends over a
4097-token window on its `hybrid_sliding` blocks, every other block.

## One expert per token, chosen beside a skip class

The router (`gate`, aliased `router`) projects the block's normed stream down to `router_hidden_size`,
adds the previous block's router state times `router_states_scale` (not on block 0), and scores
`num_experts + 1` classes with a small MLP; `router_logits` is that MLP's output, and its last column
is **skip**. The choice is the softmax plus `balancing_biases` (a buffer); the weight is the softmax's
own entry for the chosen class, not renormalized, so `expert_weights` is below one. A slot that picks
skip runs no expert: its weight is 0 and its index 0.

```python
moe = model.layers[1].mlp
with model.trace(prompt):
    logits = moe.router_logits.save()     # [batch, seq, num_experts + 1]
    w = moe.expert_weights.save()         # [batch, seq, top_k], top_k = 1
    idx = moe.expert_indices.save()

probs = logits.float().softmax(-1)
torch.testing.assert_close(probs.gather(-1, idx).to(w.dtype), w)
```

On ZAYA1-8B and ZAYA1-74B-preview `balancing_biases` is −1.0 on skip and above zero on at least one
expert of every block, so in this forward the skip class never wins a slot: its softmax mass lowers
the chosen expert's weight instead. The routing state carried to the next block means an edit to one
block's routing input reaches every later block's router.

## Ablating an expert leaves the merge's bias

With one slot per token, zeroing expert `e`'s weight removes the token's whole routed output, and
`mlp_output` there becomes `hidden_states_bias * hidden_states_scale`, not zero. Count usage on
slots with a weight, since a skipped slot reads as expert 0:

```python
moe, e = model.layers[1].mlp, 3
with model.trace(prompt):
    idx = moe.expert_indices.save()
    w = moe.expert_weights.save()
    clean = moe.mlp_output.save()
with model.trace(prompt):
    moe.expert_weights = moe.expert_weights.masked_fill(moe.expert_indices == e, 0)
    ablated = moe.mlp_output.save()

chose = ((idx == e) & (w > 0)).any(-1)                 # [batch, seq]
assert torch.equal((ablated != clean).any(-1), chose)
```

There is no shared expert, so `routed_output` is the mixture's own output, before the merge.

## The readout

`logits` is `lm_head.output`, with no cap or scale. The tokenizer is a `GemmaTokenizer` that
prepends `<bos>` (id 2).
"""
