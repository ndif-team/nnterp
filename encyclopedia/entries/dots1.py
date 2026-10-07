"""dots.llm1: Llama's pre-norm block with per-head query and key norms, one dense first block, then DeepSeek-V3's
sigmoid-routed mixture with a selection bias and two shared experts."""

MODEL_TYPE = "dots1"
TITLE = "dots.llm1"
SUBTITLE = (
    "Llama's pre-norm block with each query and key head RMS-normed, a dense MLP on block 0, and on every "
    "block after it a mixture whose experts are chosen by a sigmoid plus a selection bias that the weights "
    "leave out, beside a shared expert every token runs through."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "rednote-hilab/dots.llm1.base"
#: The tiny checkpoint the test suite builds the page from (every block a mixture of 8 experts, all 8 per token).
PINNED = "hf-tiny-v2/tiny-random-Dots1ForCausalLM"
CHECKPOINTS = ["rednote-hilab/dots.llm1.base", "rednote-hilab/dots.llm1.inst"]

#: No kin among the entries; the hash of dots1 (249) sits on nemotron_h's 250. 246 is the middle of the free
#: gap between bloom (242) and nemotron_h (250).
PALETTE = {"hue": 246}
VLLM = False
QUIRKS = ["qk-norm", "mixture-of-experts", "dense-first-blocks"]

#: Both checkpoints have 142B parameters: nothing here ran on real weights. Every identity, shape, read order and
#: snippet ran on the pinned tiny checkpoint loaded with the real routing settings (first_k_dense_replace 1,
#: num_experts_per_tok 2, norm_topk_prob true, routed_scaling_factor 2.5) and, for the bias claims, a selection
#: bias written by hand. Sizes and routing settings are the Hub configs', read on meta builds.

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
            "detail": "{num_heads} heads × {head_dim}, q/k normed",
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
    "embed": "A plain lookup: token_embeddings equals layers[0].input. The tokenizer (Qwen2's) prepends nothing.",
    "head": "lm_head has its own weight: tie_word_embeddings is false on both checkpoints. logits is lm_head.output.",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))       # q_norm, k_norm per head inside
out = h + mlp(post_attention_layernorm(h))    # dense on block 0, a mixture after
```

Llama's pre-norm block and Llama's names. `first_k_dense_replace` is 1 on both checkpoints: block
0 has a dense MLP 10944 wide, and the other 61 a mixture. Nothing norms a sublayer's output, so
`attention_output` is `o_proj`'s output and `mlp_output` the mixture's, shared expert included.
The identity is the plain sum on both kinds of block:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

## Queries and keys are normed per head

`q_norm` and `k_norm` are RMSNorms over each head's 128 dimensions, applied before the rotary;
`attention_queries` and `attention_keys` are read after both, and an edit there reaches the
attention as written. An edit to one head's slice of `q_proj.output` stays in that head, and the
norm rescales it. There are 32 key/value heads for 32 query heads, so no grouping. Every block
attends over the whole prefix (`sliding_window` is null), with rotary at `rope_theta` 1e7. The
attention interior needs `attn_implementation="eager"`.

## The router: sigmoid scores, a selection bias, weights summing to 2.5

128 routed experts, 6 per token. `router_logits` are the router's float32 logits, before the
sigmoid and before the selection bias (`router.e_score_correction_bias`). The bias is added to
the sigmoids only to choose; `n_group` and `topk_group` are 1, so every expert competes.
`expert_weights` are the chosen experts' sigmoids without the bias, renormalized to sum to one
(`norm_topk_prob`) and multiplied by `routed_scaling_factor` 2.5, so they sum to 2.5 at every token:

```python
moe = model.layers[1].mlp
with model.trace(prompt):
    logits = moe.router_logits.save()
    w = moe.expert_weights.save()
    idx = moe.expert_indices.save()

chosen = logits.sigmoid().gather(-1, idx)
torch.testing.assert_close(w, chosen / chosen.sum(-1, keepdim=True) * 2.5)
```

A written `router_logits` moves the choice and the weights; a written bias moves only the choice.

## The shared expert runs after the routed ones

`n_shared_experts` is 2, merged into one `Dots1MLP` (`mlp.shared_experts`) of width 2 × 1408. It
reads the mixture's input and runs after the routed experts, so in one trace read `routed_output`
before `shared_expert_output`. Their sum is `mlp_output`:

```python
with model.trace(prompt):
    routed = moe.routed_output.save()
    shared = moe.shared_expert_output.save()
    out = moe.mlp_output.save()

torch.testing.assert_close(routed + shared, out)
```

## Ablating an expert

Zero the weight of every slot that chose it. The token's other weights keep their values, so its
weights then sum to less than 2.5, and `mlp_output` changes on exactly the tokens that chose it:

```python
with model.trace(prompt):
    idx = moe.expert_indices.save()
    clean = moe.mlp_output.save()
with model.trace(prompt):
    moe.expert_weights = moe.expert_weights.masked_fill(moe.expert_indices == 3, 0)
    ablated = moe.mlp_output.save()

assert torch.equal((ablated != clean).any(-1), (idx == 3).any(-1))
```

Block 0 has a dense MLP: `support()` reports every mixture value missing there.
"""
