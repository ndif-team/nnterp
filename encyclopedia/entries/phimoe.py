"""Phi-3.5-MoE and the SlimMoE checkpoints that load as PhimoeForCausalLM."""

MODEL_TYPE = "phimoe"
TITLE = "Phi-3.5-MoE"
SUBTITLE = (
    "Llama's tree with LayerNorms and biased projections, and a mixture of 16 experts on every block whose "
    "router, sparsemixer, picks two experts one after the other and weights each by a softmax over the "
    "logits near that slot's maximum: a weight is exactly 1 unless another logit is within 2% of it, and a "
    "token's two weights sum to up to 2."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "microsoft/Phi-3.5-MoE-instruct"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-PhimoeForCausalLM"
CHECKPOINTS = [
    "microsoft/Phi-3.5-MoE-instruct",
    "microsoft/Phi-mini-MoE-instruct", "microsoft/Phi-tiny-MoE-instruct",
]

#: The Phi lineage: phi 106, phi3 113, phimoe 120.
PALETTE = {"hue": 120}
VLLM = False
QUIRKS = ["mixture-of-experts", "qkv-bias", "layernorm", "sliding-window"]

#: Every checkpoint is above 3B parameters, so nothing here ran on real weights: shapes, the routing arithmetic
#: and the snippets are from the pinned tiny checkpoint (16 experts, top 2) and a copy of it with
#: attention_bias and lm_head_bias set; the rotary from transformers' rotary module built on the reference
#: config; every size from the configs.

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
            "detail": "{num_heads} heads, {num_kv_heads} kv, biased",
        },
        {
            "host": "mlp",
            "kind": "moe",
            "label": "MoE",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "The MoE's input norm, as on Llama, but a LayerNorm with a bias.",
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
    "embed": "A plain lookup: token_embeddings equals layers[0].input. The tokenizer is Phi-3's and adds no BOS.",
    "norm": "A LayerNorm with a bias; project_on_vocab applies it, bias included.",
    "head": "lm_head has its own weight (tie_word_embeddings is false) and a bias (lm_head_bias). Nothing follows "
            "it: logits equals lm_head.output.",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))       # q, k, v, o projections with biases
out = h + mlp(post_attention_layernorm(h))    # 2 of 16 experts per token
```

Llama's names and Llama's order, with two differences inside: the three norms are `nn.LayerNorm`s
with a weight and a bias (their `eps` comes from `rms_norm_eps`), and on every checkpoint
`attention_bias` puts a bias on `q_proj`, `k_proj`, `v_proj` and `o_proj`. Nothing norms a
sublayer's output, so `attention_output` is `o_proj`'s output and `mlp_output` is the mixture's.
There is no shared expert: `routed_output` is `mlp_output`. The identity is the plain sum:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

## The router picks two experts one after the other

`mlp.router` is an `nn.Linear` subclass, 16 outputs, no bias; `router_logits` is its projection,
read before the scoring. `SCORING` is `"sparsemixer"`, and in eval it runs two slots in turn:

- slot 0 takes the expert with the largest logit `m`. Every expert whose logit `s` is within the
  jitter threshold, `(m - s) / max(|s|, m) <= 2 * router_jitter_noise` (0.01 on every checkpoint),
  stays in a softmax; the others are masked out. The weight is the chosen expert's share of that
  softmax;
- slot 1 removes slot 0's expert and does the same over the remaining 15.

So `expert_indices` are the top two logits, and each `expert_weights` entry is between 0 and 1,
exactly 1 when no other logit is within the threshold of that slot's maximum. The two weights are
not renormalized together: a token's weights sum to up to 2, and the mixture scales each expert's
output by its own. On the pinned checkpoint every weight is 1.0. The recomputation, checked there:

```python
moe = model.layers[1].mlp
eps = model.config.router_jitter_noise
with model.trace(prompt):
    logits = moe.router_logits.save()       # [batch, seq, num_experts]
    w = moe.expert_weights.save()           # [batch, seq, 2]
    idx = moe.expert_indices.save()

def slot(scores):                           # one slot of sparsemixer, in eval
    top = scores.argmax(-1, keepdim=True)
    m = scores.gather(-1, top)
    near = (m - logits) / logits.abs().clamp(min=m) <= 2 * eps
    return scores.masked_fill(~near, float("-inf")).softmax(-1).gather(-1, top), top

w0, i0 = slot(logits)
w1, i1 = slot(logits.scatter(-1, i0, float("-inf")))
assert torch.equal(torch.cat([i0, i1], -1), idx)
torch.testing.assert_close(torch.cat([w0, w1], -1), w)
```

The weight a near-tie takes is dropped, not passed to the other expert. Written at `router_logits`,
a last token whose logits are 1.0 on expert 2, 0.99 on expert 5 and 0.5 or less elsewhere routes to
`[2, 5]` with weights `[0.5025, 1.0]`: slot 0 shares its softmax with expert 5, which slot 1 then
takes at full weight. Under `model.train()` both slots sample their expert (Gumbel noise) and the
weights change; trace in eval.

## Ablating an expert touches only the tokens routed to it

Zeroing expert `e`'s weight where `expert_indices == e` removes that slot's term and leaves the
other slot's weight as it was. `mlp_output` then changes on exactly the tokens that chose `e`:

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

A slot whose weight is 1 contributes its expert's whole output, so removing it removes a full-weight term.
`expert_outputs` (each slot's weighted output) is served under the default
`experts_implementation`, and its sum over the two slots is `routed_output`.

## The rotary is longrope with one scale at every length

Phi-3.5-MoE sets `rope_scaling` type `longrope` over an `original_max_position_embeddings` of
4096, with `short_mscale` and `long_mscale` both 1.243: cos and sin are multiplied by 1.243 at
every position and every length, so a rotated query's norm is 1.243 times the projection's. The
transformers rotary for this class computes its frequencies from `short_factor` at every length;
`long_factor` is not used, also past 4096 tokens. Rotary covers whole 128-dimensional heads. The
`sliding_window` of 131072 equals the context and never masks; Phi-mini-MoE and Phi-tiny-MoE set
2047 with a 4096-token context and no rotary scaling.

## Load with eager for the attention interior only

The attention interior needs `attn_implementation="eager"`; there is no softcap, sink or window
short of the context on Phi-3.5-MoE, so the default `sdpa` load computes the same function. 32
query heads share 8 key/value heads of 128 dimensions: `attention_keys` and `attention_values` are
served before `repeat_kv`, so an edit to key/value head `j` reaches query heads `4j` to `4j + 3`.

## The readout has two biases

`logits` equals `lm_head.output`, and `project_on_vocab` applied to the last block's `layer_output`
equals `logits` exactly, the final LayerNorm's bias and `lm_head`'s included. A logit lens therefore
adds the same vector at every block, whatever the input:

```python
bias_logits = model.lm_head.weight @ model.norm.bias + model.lm_head.bias
```

`embed_tokens` and `lm_head` are separate matrices. The tokenizer is Phi-3's: 32011 tokens in 32064
rows, `<s>` (id 1) as `bos_token` but not prepended, so position 0 is the prompt's first token.
`' Paris'` splits as `['▁', 'Paris']`; encode `"Paris"` (id 3681) for one token.

## The checkpoints

The sizes, from the configs, each with 16 experts and top 2 on every one of 32 blocks, hidden 4096:

- Phi-3.5-MoE-instruct: 32 heads × 128 over 8, experts 6400 wide, a 131072-token context
- Phi-mini-MoE-instruct: 32 heads × 128 over 8, experts 960 wide, a 4096-token context
- Phi-tiny-MoE-instruct: 16 heads × 256 over 4, experts 448 wide, a 4096-token context

Phi-mini-MoE and Phi-tiny-MoE ship SlimMoE modeling code (`configuration_slimmoe`, `auto_map`) beside a
`model_type` of `phimoe`; without `trust_remote_code` transformers builds its own
`PhimoeForCausalLM` from the config.
"""
