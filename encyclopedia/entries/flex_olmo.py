"""FlexOlmo: OLMo-2's post-norm block with a mixture of independently trained experts in place of the MLP, every expert on every token."""

MODEL_TYPE = "flex_olmo"
TITLE = "FlexOlmo"
SUBTITLE = (
    "OLMo-2's post-norm block with a mixture of experts for the MLP: the released checkpoints route every token "
    "to every expert, weighted by a softmax over all of them, and the stream receives the post-norm of the "
    "weighted sum."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "allenai/FlexOlmo-7x7B-1T"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-FlexOlmoForCausalLM"
CHECKPOINTS = [
    "allenai/FlexOlmo-7x7B-1T", "allenai/FlexOlmo-7x7B-1T-RT",
    "allenai/Flex-math-2x7B-1T", "allenai/Flex-news-2x7B-1T", "allenai/Flex-pes2o-2x7B-1T",
    "allenai/Flex-code-2x7B-1T", "allenai/Flex-creative-2x7B-1T", "allenai/Flex-reddit-2x7B-1T",
]

#: OLMo lineage: 67 sits between olmo_hybrid's 64 and olmoe's 70.
PALETTE = {"hue": 67}
VLLM = False
QUIRKS = ["post-norms", "qk-norm", "mixture-of-experts"]

#: Every checkpoint is 2x7B or 7x7B, so no real weights were run: shapes, identities and the routing edits
#: are from the pinned tiny checkpoint (8 experts, top 2; float32, CPU), the every-expert routing also from
#: the tiny loaded with num_experts_per_tok=8; sizes, expert counts and norm_topk_prob from the checkpoints'
#: configs; the experts' training data from the model cards.

BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "post_norm": "post_attention_layernorm",
            "post_norm_note": "On FlexOlmo this norm follows the attention and its output is attention_output. "
                              "On Llama the same name is the norm before the MLP; FlexOlmo has no norm before "
                              "either sublayer.",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} heads × {head_dim}, q/k normed",
        },
        {
            "host": "mlp",
            "kind": "moe",
            "label": "MoE",
            "post_norm": "post_feedforward_layernorm",
            "post_norm_note": "Norms the mixture's weighted sum; its output is mlp_output.",
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
    "embed": "A plain lookup: token_embeddings equals layers[0].input. The tokenizer prepends no BOS.",
    "norm": "A plain RMSNorm: the gain is norm.weight (not 1 + weight), eps 1e-6. project_on_vocab applies it.",
    "head": "lm_head has its own weight: tie_word_embeddings is false on every checkpoint. Nothing follows it: "
            "logits equals lm_head.output.",
}

NOTES = """
## The block, in order

```
h   = x + post_attention_layernorm(self_attn(x))
out = h + post_feedforward_layernorm(mlp(h))     # mlp: router, experts, weighted sum
```

OLMo-2's block with the MLP replaced by a mixture on every block. Two RMSNorms, both after a
sublayer, none before: `self_attn.input` is `layers[i].input` and `mlp.input` is
`layers[i].input + attention_output`, both unnormalized. `post_attention_layernorm` follows the
attention, where on Llama the same name is the MLP's input norm.

## The contributions are the post-norms' outputs

`attention_output` is `post_attention_layernorm`'s output and `mlp_output` is
`post_feedforward_layernorm`'s. The mixture's own output is `routed_output`, which equals
`mlp.output` (there is no shared expert), and `mlp_output` is the post-norm of it. The identity is
the plain sum, exact in float32:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    routed = model.layers[1].mlp.routed_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

The post-norm divides by the RMS of what it reads, so a uniform scale of `routed_output`, or of every
`expert_weights` entry of a token together, cancels in `mlp_output` except through `eps` (1e-6). What
the routing decides is the mix of the experts' directions, not the size of the term. Scale
`mlp_output` to scale what the block adds.

## Every expert runs on every token

`num_experts_per_tok` equals `num_experts` on every released checkpoint: 7 of 7 on FlexOlmo-7x7B-1T
and -RT, 2 of 2 on each Flex-…-2x7B-1T. The router takes a softmax over all logits in float32 and
keeps the top `top_k`; `norm_topk_prob` is false, but with every expert kept the weights are the
whole softmax and sum to one. `expert_indices` is then every expert, ordered by weight, slot 0 the
largest; which slot an expert takes changes from token to token. The pinned tiny checkpoint keeps 2
of 8, so there the weights sum to less than one.

```python
moe = model.layers[1].mlp
with model.trace(prompt):
    logits = moe.router_logits.save()       # [batch, seq, num_experts]
    w = moe.expert_weights.save()           # [batch, seq, top_k]
    idx = moe.expert_indices.save()

top = logits.float().softmax(-1).topk(moe.top_k, dim=-1)
torch.testing.assert_close(top.values.to(w.dtype), w)    # the softmax's own entries
assert torch.equal(top.indices, idx)
```

## The experts are trained on separate data

Per the model cards, FlexOlmo-7x7B-1T combines a public-mix expert, trained on 1T tokens, with six
experts branched from it and trained on 50B tokens each of news, math, code, academic text, creative
writing and Reddit; FlexOlmo-7x7B-1T-RT has the same configuration. Each
Flex-…-2x7B-1T pairs the public-mix expert with one domain expert. The configs do not say which index
holds which expert. Each expert is a SwiGLU MLP `intermediate_size` (11008) wide.

## Removing an expert: the weight or the logit

Zeroing expert `e`'s weight where `expert_indices == e` removes its term and leaves the others'
weights as they were, so a token's weights then sum to `1 - w_e`. With every expert on every token,
that changes every token. Writing `-inf` at `router_logits` for `e` instead gives `e` a weight of
exactly 0 and renormalizes the softmax over the rest, as if the router had no logit for `e`:

```python
moe, e = model.layers[1].mlp, 3
with model.trace(prompt):
    moe.router_logits[..., e] = float("-inf")    # weights renormalized over the other experts
    w = moe.expert_weights.save()

with model.trace(prompt):
    moe.expert_weights = moe.expert_weights.masked_fill(moe.expert_indices == e, 0)   # not renormalized
    w_zeroed = moe.expert_weights.save()
```

With every expert kept, the two give `routed_output`s that differ by the factor `1 / (1 - w_e)` per
token, which the post-norm divides out: the two edits give the same `mlp_output` up to `eps`. On a
checkpoint that keeps fewer than all experts (the pinned tiny one) they differ, since `-inf` lets
another expert into the top-k.

## Queries and keys are normed across all heads

`q_norm` and `k_norm` are RMSNorms over the whole 4096-wide projection, applied before the heads
are split and before the rotary, and `attention_queries` and `attention_keys` are read after both.
An edit to one head's slice of `q_proj.output` reaches the other heads through the shared RMS; edit
a single head at `attention_queries`. Every checkpoint has 32 key/value heads for 32 query heads.

## Loading

The attention interior needs `attn_implementation="eager"`; `expert_outputs` needs the default
`experts_implementation`. There is no softcap, sink or window; the rotary is the plain one,
`rope_theta` 500000. The weights are stored in float32 (`dtype` in the config): 133 GB for 7x7B,
47 GB for each 2x7B.

## The readout and the tokenizer

`logits` is `lm_head.output`, with no cap or scale, and `project_on_vocab` on the last block's
`layer_output` equals `logits`. `lm_head` and `embed_tokens` are separate weights. The tokenizer is
`dolma2`: it prepends nothing, so `model.input_ids` is the prompt's tokens alone, and
`<|endoftext|>` (id 100257) is both `bos_token` and `eos_token`.
"""
