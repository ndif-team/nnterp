"""ERNIE 4.5 MoE: ERNIE 4.5's block with dense first blocks, then a softmax-routed mixture whose choice adds a
correction bias to the probabilities, and on 21B-A3B a shared expert that runs before the router."""

MODEL_TYPE = "ernie4_5_moe"
TITLE = "ERNIE 4.5 MoE"
SUBTITLE = (
    "Llama's pre-norm block with rotary on adjacent pairs of dimensions, a dense MLP on the first blocks and a "
    "mixture after them, whose experts are chosen by a softmax plus a correction bias that the weights leave "
    "out, renormalized to sum to one."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "baidu/ERNIE-4.5-21B-A3B-Base-PT"
#: The tiny checkpoint the test suite builds the page from (both blocks a mixture of 8 experts, top 2, 2 shared).
PINNED = "hf-tiny-v2/tiny-random-Ernie4_5_MoeForCausalLM"
#: The 300B-A47B checkpoints' mixture has no shared expert: their page draws no shared panel.
CHECKPOINTS = [
    "baidu/ERNIE-4.5-21B-A3B-Base-PT", "baidu/ERNIE-4.5-21B-A3B-PT", "baidu/ERNIE-4.5-21B-A3B-Thinking",
    "baidu/ERNIE-4.5-300B-A47B-Base-PT", "baidu/ERNIE-4.5-300B-A47B-PT",
]

#: ERNIE lineage: ernie4_5 sets 180; 185 sits beside it, short of cohere2's 188.
PALETTE = {"hue": 185}
VLLM = False
QUIRKS = ["interleaved-rotary", "mixture-of-experts", "dense-first-blocks"]

#: The smallest checkpoint has 21B parameters: nothing here ran on real weights. Every identity, shape, read
#: order and snippet ran on the pinned tiny checkpoint, loaded with moe_layer_start_index 1 for the dense-block
#: claims and with a correction bias written by hand for the bias claims. Sizes and block layouts are the Hub
#: configs', read on meta builds.

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
            "detail": "{num_heads} heads × {head_dim}, {num_kv_heads} kv",
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
    "embed": "A plain lookup: token_embeddings equals layers[0].input. The tokenizer prepends nothing; <s> (id 1) "
             "is its BOS.",
    "layers": "The configs' multi-token-prediction block (num_nextn_predict_layers) is not built: layers holds "
              "num_hidden_layers blocks.",
    "head": "lm_head shares its weight with embed_tokens on 21B-A3B (tie_word_embeddings) and has its own on "
            "300B-A47B. logits is lm_head.output.",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))
out = h + mlp(post_attention_layernorm(h))    # dense first, a mixture after
```

Llama's pre-norm block and Llama's names. A block has a mixture when its index is between
`moe_layer_start_index` and `moe_layer_end_index` and a multiple of `moe_layer_interval` (1) in
one-based counting; every other block has a dense MLP `intermediate_size` wide. On 21B-A3B block 0
is dense and blocks 1 to 27 are mixtures; on 300B-A47B blocks 0 to 2 are dense and 3 to 53 mixtures.
Nothing norms a sublayer's output, so the identity is the plain sum on both kinds:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

## Attention

20 query heads over 4 key/value heads on 21B-A3B, 64 over 8 on 300B-A47B, `head_dim` 128 on both.
The rotary turns adjacent pairs of dimensions (2i, 2i + 1) of each query and key head, as on the
dense ERNIE 4.5, so `attention_queries` keeps `q_proj`'s dimension order: at position 0 it is
`q_proj`'s output split into heads. No projection has a bias. The attention interior needs
`attn_implementation="eager"`.

## The router: a softmax, a correction bias, weights summing to one

64 routed experts on every checkpoint, 6 per token on 21B-A3B (`moe_k`) and 8 on 300B-A47B.
`router_logits` are float32 logits over all 64. The router takes their softmax, adds the
correction bias (`router.moe_statics.e_score_correction_bias`) to the probabilities to choose the
experts, and returns the chosen experts' probabilities without the bias, renormalized to sum to
one (the sum is clamped at `moe_norm_min`, 1e-12):

```python
moe = model.layers[1].mlp
with model.trace(prompt):
    logits = moe.router_logits.save()
    w = moe.expert_weights.save()
    idx = moe.expert_indices.save()

chosen = logits.float().softmax(-1).gather(-1, idx)
torch.testing.assert_close(w, (chosen / chosen.sum(-1, keepdim=True)).to(w.dtype))
```

A written `router_logits` moves the choice and the weights; a written bias moves only the choice.
The bias is added to probabilities, so a bias of 0.1 outweighs any gap between two probabilities
smaller than 0.1.

## The shared expert runs before the router

21B-A3B has `moe_num_shared_experts` 2, merged into one `Ernie4_5_MoeMLP` (`mlp.shared_experts`)
of width 2 × 1536. It runs on every token before the router, so in one trace read
`shared_expert_output` before `router_logits`. `routed_output + shared_expert_output` is
`mlp_output`:

```python
with model.trace(prompt):
    shared = moe.shared_expert_output.save()
    routed = moe.routed_output.save()
    out = moe.mlp_output.save()

torch.testing.assert_close(routed + shared, out)
```

300B-A47B has no shared expert (`moe_num_shared_experts` 0): `shared_expert_output` is
unavailable there, and `routed_output` is `mlp_output`.

## Ablating an expert

Zero the weight of every slot that chose it. The token's other weights keep their values, so they
then sum to less than one, and `mlp_output` changes on exactly the tokens that chose it:

```python
with model.trace(prompt):
    idx = moe.expert_indices.save()
    clean = moe.mlp_output.save()
with model.trace(prompt):
    moe.expert_weights = moe.expert_weights.masked_fill(moe.expert_indices == 3, 0)
    ablated = moe.mlp_output.save()

assert torch.equal((ablated != clean).any(-1), (idx == 3).any(-1))
```

The dense blocks have no router: `support()` reports every mixture value missing there.
"""
