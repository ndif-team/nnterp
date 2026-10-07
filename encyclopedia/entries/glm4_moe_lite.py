"""GLM-4.7-Flash: DeepSeek-V2's latent attention on Llama's block, a dense first block and a sigmoid-routed mixture after."""

MODEL_TYPE = "glm4_moe_lite"
TITLE = "GLM-4.7-Flash"
SUBTITLE = (
    "Latent attention, whose keys and values are expanded for every head from one latent per token, "
    "and a mixture of 64 experts and a shared one, chosen by a sigmoid plus a selection bias, after a dense first block."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "zai-org/GLM-4.7-Flash"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-Glm4MoeLiteForCausalLM"
#: The one public checkpoint whose config is glm4_moe_lite (GLM-4.7 itself is glm4_moe).
CHECKPOINTS = ["zai-org/GLM-4.7-Flash"]

#: GLM lineage: glm sets 84 (glm4 79, glm4_moe 89); the latent-attention line sits at 94 (glm_moe_dsa 74).
PALETTE = {"hue": 94}
VLLM = False
QUIRKS = ["latent-attention", "interleaved-rotary", "mixture-of-experts", "dense-first-blocks"]

#: Shapes, identities, read orders, the rotary order and the routing were run on the pinned tiny checkpoint
#: (block 0 dense, block 1 a mixture of 8 experts, top 2; queries, keys and values 128 wide). Sizes, scales,
#: the routing settings and the tokenizer are GLM-4.7-Flash's, read off its config, a meta build and its
#: tokenizer; no real weights were run (the checkpoint has 30B parameters).

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
    "embed": "A plain lookup: token_embeddings equals layers[0].input. The tokenizer prepends nothing.",
    "layers": "The checkpoint's multi-token-prediction block (num_nextn_predict_layers) is not built: layers holds num_hidden_layers blocks.",
    "head": "lm_head has its own weight (tie_word_embeddings is false); logits is lm_head.output.",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))      # multi-head latent attention
out = h + mlp(post_attention_layernorm(h))   # dense on block 0, a mixture after
```

Llama's pre-norm block and Llama's names. `config.mlp_layer_types` says which blocks are dense:
on GLM-4.7-Flash block 0 has a dense MLP and the other 46 a mixture. The block adds each
sublayer's output to the stream and returns a tensor, so the identity is the plain sum on a dense
block and on a mixture block:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

The config carries one multi-token-prediction block (`num_nextn_predict_layers` 1), stored after
the last block; transformers does not build it, so `model.layers` has 47 blocks.

## Latent attention: every head's key comes from one latent

`attention_queries` and `attention_keys` are `qk_head_dim` wide (`qk_nope_head_dim +
qk_rope_head_dim`, 192 + 64 = 256), `attention_values` and `attention_head_outputs` `head_dim`
(`v_head_dim`, also 256 on GLM-4.7-Flash). The config's `head_dim` is an alias of
`qk_rope_head_dim`, which no served value has. Queries go through a compressed rank (`q_lora_rank`
768: `q_a_proj`, `q_a_layernorm`, `q_b_proj`, and `self_attn.q_proj` is `None`). Keys and values
are expanded from one latent per token (`kv_lora_rank` 512) for every head, so `attention_keys`
has `num_heads` (20) heads, and the last 64 dimensions of every head's key are one shared rotary
key. The attention interior needs `attn_implementation="eager"`.

```python
attn = model.layers[1].self_attn
with model.trace(prompt):
    q = attn.attention_queries.save()          # [1, 20, seq, 256]
    k = attn.attention_keys.save()             # [1, 20, seq, 256]
    v = attn.attention_values.save()           # [1, 20, seq, 256]

assert torch.equal(k[:, 0, :, -64:], k[:, 1, :, -64:])   # one rotary key
```

The scores are scaled by `qk_head_dim ** -0.5` (`self_attn.scaling`, 0.0625): the config has no
`rope_scaling`.

## The rotary part is served de-interleaved

`rope_interleave` is true (the config class's default; the checkpoint's config does not set it):
the attention rotates pair (2i, 2i + 1) of the projection into dimensions i and i + 32 of the
rotary part. `attention_queries[..., -64:]` and `attention_keys[..., -64:]` are in that
rotate-half order, not the order of `q_b_proj`'s output: at position 0, where rotary is the
identity, the served rotary part is the projection's even dimensions followed by its odd ones.

```python
with model.trace(prompt):
    proj = attn.q_b_proj.output.save()
with model.trace(prompt):
    q = attn.attention_queries.save()

rope = proj.view(1, -1, model.num_heads, model.qk_head_dim)[0, 0, 0, -64:]
torch.testing.assert_close(q[0, 0, 0, -64:], torch.cat([rope[0::2], rope[1::2]]))
```

## The mixture: sigmoid scores, a selection bias, renormalized weights

64 routed experts, 4 per token, and one shared expert every token runs through.
`router_logits` are the logits before the sigmoid, and before the selection bias
(`router.e_score_correction_bias`). The bias is added to the sigmoids only to choose; `n_group`
and `topk_group` are 1, so every expert is a candidate. `expert_weights` are the chosen experts'
sigmoids without the bias, renormalized to sum to one (`norm_topk_prob`) and multiplied by
`routed_scaling_factor`, so they sum to 1.8 at every token. A written `router_logits` moves both
the choice and the weights; a written bias moves only the choice.

```python
moe = model.layers[1].mlp
with model.trace(prompt):
    logits = moe.router_logits.save()
    w = moe.expert_weights.save()
    idx = moe.expert_indices.save()
    routed = moe.routed_output.save()
    shared = moe.shared_expert_output.save()
    out = moe.mlp_output.save()

chosen = logits.sigmoid().gather(-1, idx)
scale = model.config.routed_scaling_factor
torch.testing.assert_close(w, chosen / chosen.sum(-1, keepdim=True) * scale)
torch.testing.assert_close(routed + shared, out)
```

The shared expert runs after the routed ones, so `shared_expert_output` is read after
`routed_output`. Ablating an expert is zeroing the weight of every slot that chose it; the other
slots keep their weights, so the token's weights then sum to less than 1.8:

```python
with model.trace(prompt):
    moe.expert_weights = moe.expert_weights.masked_fill(moe.expert_indices == 3, 0)
    ablated = model.logits.save()
```

Block 0 has a dense MLP: `support()` reports every mixture value missing there.
"""
