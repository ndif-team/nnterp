"""DeepSeek-V2: multi-head latent attention, a dense first block and a softmax-routed mixture with shared experts after."""

MODEL_TYPE = "deepseek_v2"
TITLE = "DeepSeek-V2 / DeepSeek-Coder-V2"
SUBTITLE = (
    "Latent attention, whose queries and keys are wider than its values and share one rotary key across "
    "heads, and a softmax-routed mixture with two shared experts on every block but the first."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "deepseek-ai/DeepSeek-V2-Lite"
#: The tiny checkpoint the test suite builds the page from (a mixture on both of its blocks).
PINNED = "hf-tiny-v2/tiny-random-DeepseekV2ForCausalLM"
CHECKPOINTS = [
    "deepseek-ai/DeepSeek-V2-Lite", "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "deepseek-ai/DeepSeek-V2", "deepseek-ai/DeepSeek-V2-Chat", "deepseek-ai/DeepSeek-V2-Chat-0628",
    "deepseek-ai/DeepSeek-V2.5", "deepseek-ai/DeepSeek-V2.5-1210",
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Base", "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "deepseek-ai/DeepSeek-Coder-V2-Base", "deepseek-ai/DeepSeek-Coder-V2-Instruct",
    "deepseek-ai/DeepSeek-Coder-V2-Instruct-0724",
]

#: DeepSeek lineage: set here; deepseek_v3 sits at 2. Every hue from 99 to 127 is within 15° of
#: gpt_neox (99) or gemma4_unified_text (127), so the lineage sits between kimi_linear (342) and mamba (17).
PALETTE = {"hue": 357}
VLLM = True
QUIRKS = ["latent-attention", "interleaved-rotary", "mixture-of-experts", "dense-first-blocks"]

#: Shapes, identities and read orders were run on the pinned tiny checkpoint and on a random model built
#: from DeepSeek-V2-Lite's config with smaller widths (q_lora_rank null, queries and keys 24 wide, values
#: 16), whose dense block 0 the pinned tiny lacks. Sizes, scales and routing settings are the Hub configs',
#: read on meta builds; no real weights were run (the smallest checkpoint, V2-Lite, has 16B parameters).

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
            "detail": "latent, {num_heads} heads, q·k {qk_head_dim}, v {v_head_dim}",
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
    "embed": "A plain lookup: token_embeddings equals layers[0].input.",
    "head": "lm_head has its own weight (tie_word_embeddings is false); logits is lm_head.output.",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))      # multi-head latent attention
out = h + mlp(post_attention_layernorm(h))   # dense on block 0, a mixture after
```

Llama's pre-norm block and Llama's names. `first_k_dense_replace` is 1 on every released
checkpoint: block 0's `mlp` is a dense MLP, every later block's a mixture. The block adds each
sublayer's output to the stream and returns a tensor, so the identity is the plain sum, on the
dense block as on a mixture block:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

## Latent attention: queries and keys are wider than values

`attention_queries` and `attention_keys` are `[batch, heads, seq, qk_head_dim]`, with
`qk_head_dim = qk_nope_head_dim + qk_rope_head_dim` (128 + 64 = 192); `attention_values` and
`attention_head_outputs` are `head_dim` wide, which nnterp reads from `v_head_dim` (128). The
config's own `head_dim` key is `qk_rope_head_dim`, the rotary width, which no served value has.
The attention interior needs `attn_implementation="eager"`.

```python
attn = model.layers[1].self_attn
with model.trace(prompt):
    q = attn.attention_queries.save()          # [1, 16, seq, 192] on V2-Lite
    v = attn.attention_values.save()           # [1, 16, seq, 128]
```

Keys and values are expanded from one compressed latent per token (`kv_lora_rank`, 512) by
`kv_b_proj`, so `attention_keys` has `num_heads` heads, as many as the queries. The last
`qk_rope_head_dim` dimensions of a key come from one projection per token, shared by every head:
`attention_keys[:, h, :, -64:]` is the same tensor for every `h`, and an edit there has to be made
on every head to act like an edit to the shared part.

## Query compression differs between the sizes

DeepSeek-V2, V2.5 and Coder-V2 (236B) compress the queries too (`q_lora_rank` 1536):
`q_a_proj`, `q_a_layernorm`, `q_b_proj`, and `self_attn.q_proj` is `None`. V2-Lite and
Coder-V2-Lite (16B) have `q_lora_rank` null: one `q_proj`, and no `q_a_proj`, `q_a_layernorm` or
`q_b_proj`. `attention_queries` is read at the attention interface on both, the same width.

## Rotary turns adjacent pairs, and the query scale carries YaRN's factor

The rotary part is rotated as complex numbers over adjacent pairs of dimensions (2i, 2i + 1), in
place, so the last 64 dimensions of `attention_queries` keep the order of the projection's output.
Every released checkpoint extends the context with YaRN (`rope_scaling` factor 40,
`mscale_all_dim` 0.707), and the attention multiplies the scores by `qk_head_dim ** -0.5` times
1.59 (`self_attn.scaling`, 0.1147), not by `qk_head_dim ** -0.5` alone: recompute
`attention_scores` from the queries and keys with that scale.

## The mixture: a softmax router whose weights are not renormalized

Every block after the first routes each token to `num_experts_per_tok` (6) of `n_routed_experts`
experts (64 on Lite, 160 on the 236B models) and runs it through `n_shared_experts` (2) shared
experts, one MLP `2 × moe_intermediate_size` wide. `router_logits` are the logits before the
softmax. `expert_weights` are the chosen experts' softmax probabilities times
`routed_scaling_factor`, never renormalized over the chosen six: on the Lite models the factor is
1.0 and a token's weights sum to less than one; on the 236B models it is 16.0. The 236B models
route `group_limited_greedy`: the experts sit in `n_group` 8 groups, each group scored by its best
expert, and a token chooses its six among the experts of its best `topk_group` 3 groups; the Lite
models route `greedy`, over every expert.

```python
moe = model.layers[1].mlp
with model.trace(prompt):
    logits = moe.router_logits.save()
    w = moe.expert_weights.save()
    idx = moe.expert_indices.save()
    routed = moe.routed_output.save()
    shared = moe.shared_expert_output.save()
    out = moe.mlp_output.save()

scale = model.config.routed_scaling_factor
torch.testing.assert_close(w, logits.softmax(-1).gather(-1, idx) * scale)   # greedy routing
torch.testing.assert_close(routed + shared, out)
```

The shared experts run after the routed ones. Ablating an expert is zeroing the weight of every
slot that chose it; the shared experts are untouched:

```python
with model.trace(prompt):
    moe.expert_weights = moe.expert_weights.masked_fill(moe.expert_indices == 3, 0)
    ablated = model.logits.save()
```

Block 0's MLP is dense: `support()` reports every mixture value missing there.
"""
