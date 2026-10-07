"""AFMoE (Arcee's Trinity): a sandwich block, gated attention with rotary on its sliding blocks only, dense first
blocks, then a sigmoid-routed mixture with a shared expert."""

MODEL_TYPE = "afmoe"
TITLE = "Trinity Nano / Trinity Mini / Trinity Large"
SUBTITLE = (
    "A sandwich block whose attention gates its heads with a sigmoid and turns queries and keys by rotary only "
    "on its sliding-window blocks, and whose MLP is dense on the first blocks and after them a mixture with a "
    "shared expert, chosen by a sigmoid plus a selection bias."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "arcee-ai/Trinity-Nano-Base"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-AfmoeForCausalLM"
CHECKPOINTS = [
    "arcee-ai/Trinity-Nano-Base", "arcee-ai/Trinity-Nano-Base-Pre-Anneal", "arcee-ai/Trinity-Nano-Preview",
    "arcee-ai/Trinity-Mini-Base", "arcee-ai/Trinity-Mini-Base-Pre-Anneal", "arcee-ai/Trinity-Mini",
    "arcee-ai/Trinity-Large-TrueBase", "arcee-ai/Trinity-Large-Base", "arcee-ai/Trinity-Large-Preview",
    "arcee-ai/Trinity-Large-Thinking",
]

#: Set by hues.py (no kin, in a gap between lineages).
PALETTE = {"hue": 230}
VLLM = False
QUIRKS = [
    "sandwich-norms", "embedding-multiplier", "qk-norm", "output-gate", "sliding-window", "nope-blocks",
    "mixture-of-experts", "dense-first-blocks",
]

#: The smallest checkpoint, Trinity Nano, has about 6B parameters, so nothing here ran on real weights. Every
#: identity, shape, read order and snippet ran on the pinned tiny checkpoint (block 0 sliding and dense,
#: block 1 full and a mixture of 4 experts, top 2, two shared experts' width), with the config overridden at
#: load where a note says so (mup_enabled, sliding_window, a hand-written expert_bias). Sizes, scales and
#: block layouts are the Hub configs', read on meta builds.

BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "post_norm": "post_attention_layernorm",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} heads, {num_kv_heads} kv, q/k normed, gated",
            "variants": {
                "sliding_attention": "window {sliding_window}, rotary, gated",
                "full_attention": "full causal, no rotary, gated",
            },
            "post_norm_note": "On AFMoE this norm follows the attention. On Llama the same name is the norm before the MLP.",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "pre_mlp_layernorm",
            "post_norm": "post_mlp_layernorm",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
        {
            "host": "mlp",
            "kind": "moe",
            "part_order": ["router", "shared", "experts"],
            "label": "MoE",
            "pre_norm": "pre_mlp_layernorm",
            "post_norm": "post_mlp_layernorm",
            "contribution": "mlp_output",
            "interior": [
                "router_logits", "expert_weights", "expert_indices",
                "expert_outputs", "routed_output", "shared_expert_output",
            ],
            "detail": "{num_experts} experts × {moe_intermediate_size}, top {top_k}",
            "post_norm_note": "It norms the shared and routed experts' sum: mlp_output is this norm's output, not routed_output + shared_expert_output.",
        },
    ],
}

STRIP = {
    "embed": "With mup_enabled (true on every checkpoint) the model multiplies the embedding by √hidden_size "
             "before block 0, outside embed_tokens: token_embeddings is unscaled, layers[0].input is "
             "token_embeddings · √hidden_size (32 on Nano). The tokenizer prepends <|begin_of_text|>.",
    "head": "lm_head has its own weight: tie_word_embeddings is false on every checkpoint. logits is lm_head.output.",
}

NOTES = """
## The block, in order

```
h   = x + post_attention_layernorm(self_attn(input_layernorm(x)))
out = h + post_mlp_layernorm(mlp(pre_mlp_layernorm(h)))   # dense, then a mixture
```

Four RMSNorms per block, two per sublayer. `post_attention_layernorm` *follows*
the attention, where Llama's same name is the norm before the MLP; the MLP's input norm is
`pre_mlp_layernorm`. `num_dense_layers` blocks come first with a dense MLP (2 on Nano and Mini,
6 on Large), and every block after them has a mixture. The identity is the plain sum on both kinds:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

## The contributions are the post-norms' outputs

`attention_output` is `post_attention_layernorm`'s output and `mlp_output` is
`post_mlp_layernorm`'s; `self_attn.output[0]` and `mlp.output` are the tensors entering them.
RMSNorm is scale-invariant up to `eps`, so scaling `mlp.output` leaves the stream as it was:
scale or steer the contribution itself.

```python
with model.trace(prompt):
    model.layers[1].mlp.mlp_output[:] *= 0.5     # halves what the block adds
```

The same holds inside the mixture: `routed_output + shared_expert_output` is `mlp.output`, the
tensor entering `post_mlp_layernorm`, so an edit to an expert reaches the stream after the norm
has rescaled the token's whole MLP term.

## Attention: per-head q/k norms and a sigmoid gate on every head

`q_norm` and `k_norm` are RMSNorms over each head's 128 dimensions, applied before the rotary;
`attention_queries` and `attention_keys` are read after both. The attention projects a gate from
its input (`gate_proj`, one value per head dimension) and multiplies the heads' outputs by its
sigmoid before `o_proj`. `attention_head_outputs` are the heads' outputs before the gate:

```python
attn = model.layers[1].self_attn
with model.trace(prompt):
    gate = attn.gate_proj.output.save()
with model.trace(prompt):
    heads = attn.attention_head_outputs.save()    # [batch, seq, heads, head_dim], ungated
    o_in = attn.o_proj.input.save()

torch.testing.assert_close(o_in, heads.flatten(-2) * gate.sigmoid())
```

The heads are grouped: 8 query heads over 2 key/value heads on Nano, 32 over 4 on Mini and 48
over 8 on Large. The attention interior needs `attn_implementation="eager"`; the default load is
`sdpa`.

## Rotary runs on the sliding-window blocks only

Every fourth block attends over the whole prefix (`global_attn_every_n_layers` 4: blocks 3, 7,
11, ...); the others attend over the last `sliding_window` tokens (2048 on Nano and Mini, 4096 on
Large). Only the sliding blocks turn queries and keys by rotary. On a full block nothing carries
position into the attention but the causal mask, and `attention_queries` is `q_norm`'s output as it is:

```python
attn = model.layers[1].self_attn                  # a full block on the pinned tiny
with model.trace(prompt):
    qn = attn.q_norm.output.save()                # [batch, seq, heads, head_dim]
with model.trace(prompt):
    q = attn.attention_queries.save()             # [batch, heads, seq, head_dim]

torch.testing.assert_close(q, qn.transpose(1, 2))
```

## The router: sigmoid scores, a selection bias, weights summing to route_scale

128 routed experts, 8 per token, on Nano and Mini; 256, 4 per token, on Large. `router_logits`
are `router.gate`'s output, in float32, before the sigmoid and before the bias. The bias is the
mixture's own buffer, `mlp.expert_bias`, added to the sigmoids only to choose the experts; there
are no expert groups. `expert_weights` are the chosen experts' sigmoids without the bias,
renormalized to sum to one and multiplied by `route_scale`, so they sum to 2.826 at every token
on Nano and Mini and 2.448 on Large:

```python
moe = model.layers[1].mlp
with model.trace(prompt):
    logits = moe.router_logits.save()
    w = moe.expert_weights.save()
    idx = moe.expert_indices.save()

chosen = logits.sigmoid().gather(-1, idx)
scale = model.config.route_scale
torch.testing.assert_close(w, chosen / chosen.sum(-1, keepdim=True) * scale)
```

A written `router_logits` moves the choice and the weights; a written `expert_bias` moves only the
choice.

## The shared expert runs between the router and the routed experts

One shared expert (`mlp.shared_experts`, an `AfmoeMLP` of width `moe_intermediate_size` ×
`num_shared_experts`) runs on every token after the router and before the routed experts, and the
mixture returns their sum. In one trace read `router_logits`, then `shared_expert_output`, then
`expert_weights` and `expert_indices` (they are the routed experts' arguments), then
`expert_outputs` and `routed_output`:

```python
with model.trace(prompt):
    logits = moe.router_logits.save()
    shared = moe.shared_expert_output.save()
    w = moe.expert_weights.save()
    routed = moe.routed_output.save()
    total = moe.output.save()

torch.testing.assert_close(routed + shared, total)
```

The shared expert is an `Mlp` too, and its `mlp_output` is unavailable: the block adds the
mixture's post-normed sum, at `layers[i].mlp`.

## Ablating an expert

Zero the weight of every slot that chose it. The token's other weights keep their values, so its
weights then sum to less than `route_scale`:

```python
with model.trace(prompt):
    moe.expert_weights = moe.expert_weights.masked_fill(moe.expert_indices == 3, 0)
    ablated = model.logits.save()
```

`routed_output` changes on exactly the tokens that chose the expert. The dense first blocks have
no router: `support()` reports every mixture value missing there.
"""
