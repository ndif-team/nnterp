"""DeepSeek-V3: V2's latent attention, three dense first blocks, and a sigmoid-routed mixture with a selection bias."""

MODEL_TYPE = "deepseek_v3"
TITLE = "DeepSeek-V3 / DeepSeek-R1 / Moonlight"
SUBTITLE = (
    "Latent attention, whose queries and keys are wider than its values, and a mixture with a shared "
    "expert whose experts are chosen by a sigmoid plus a selection bias that the weights leave out."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "deepseek-ai/DeepSeek-V3"
#: The tiny checkpoint; the suite rewrites its config (first_k_dense_replace 1, topk_group 1) into a local copy.
PINNED = "hf-internal-testing/tiny-random-DeepseekV3ForCausalLM"
CHECKPOINTS = [
    "deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-V3-Base", "deepseek-ai/DeepSeek-V3-0324",
    "deepseek-ai/DeepSeek-R1", "deepseek-ai/DeepSeek-R1-Zero", "deepseek-ai/DeepSeek-R1-0528",
    "deepseek-ai/DeepSeek-V3.1", "deepseek-ai/DeepSeek-V3.1-Base", "deepseek-ai/DeepSeek-V3.1-Terminus",
    "moonshotai/Moonlight-16B-A3B", "moonshotai/Moonlight-16B-A3B-Instruct",
]

#: Set by hues.py (lineage: DeepSeek).
PALETTE = {"hue": 325}
VLLM = True
QUIRKS = ["latent-attention", "mixture-of-experts", "dense-first-blocks"]

#: Shapes, identities, read orders and the routing were run on the suite's copy of the pinned tiny checkpoint
#: (block 0 dense, block 1 a mixture of 4 experts, top 2; queries and keys 192 wide, values 128), with a
#: selection bias written by hand for the bias claims. Sizes, scales and routing settings are the Hub
#: configs', read on meta builds; no real weights were run (the smallest checkpoint, Moonlight, has 16B).

#: The sublayers in forward order; a block draws the MLP its ``mlp`` is (dense on blocks 0-2, a mixture after).
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
    "layers": "The checkpoints' multi-token-prediction block (num_nextn_predict_layers) is not built: layers holds num_hidden_layers blocks.",
    "head": "lm_head has its own weight (tie_word_embeddings is false); logits is lm_head.output.",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))      # multi-head latent attention
out = h + mlp(post_attention_layernorm(h))   # dense on blocks 0-2, a mixture after
```

Llama's pre-norm block and Llama's names. `first_k_dense_replace` is 3 on the DeepSeek
checkpoints (1 on Moonlight): blocks 0 to 2 have a dense MLP, the other 58 a mixture. The block
adds each sublayer's output to the stream and returns a tensor, so the identity is the plain sum
on a dense block and on a mixture block:

```python
with model.trace(prompt):
    x = model.layers[3].input.save()
    attn = model.layers[3].self_attn.attention_output.save()
    mlp = model.layers[3].mlp.mlp_output.save()
    out = model.layers[3].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

The DeepSeek configs carry one multi-token-prediction block (`num_nextn_predict_layers` 1),
stored after the last block; transformers does not build it, so `model.layers` has 61 blocks.

## Latent attention: queries and keys are wider than values

`attention_queries` and `attention_keys` are `qk_head_dim` wide (`qk_nope_head_dim +
qk_rope_head_dim`, 128 + 64 = 192), `attention_values` and `attention_head_outputs` `head_dim`
(`v_head_dim`, 128). The config's own `head_dim` key is `qk_rope_head_dim`, which no served value
has. Queries go through a compressed rank (`q_lora_rank` 1536: `q_a_proj`, `q_a_layernorm`,
`q_b_proj`, and `self_attn.q_proj` is `None`) on every DeepSeek checkpoint; Moonlight has
`q_lora_rank` null and one `q_proj`. Keys and values are expanded from one latent per token
(`kv_lora_rank` 512) for every head, so `attention_keys` has `num_heads` (128) heads, and the
last 64 dimensions of every head's key are one shared rotary key. The attention interior needs
`attn_implementation="eager"`.

```python
attn = model.layers[3].self_attn
with model.trace(prompt):
    q = attn.attention_queries.save()          # [1, 128, seq, 192]
    k = attn.attention_keys.save()             # [1, 128, seq, 192]
    v = attn.attention_values.save()           # [1, 128, seq, 128]

assert torch.equal(k[:, 0, :, -64:], k[:, 1, :, -64:])   # one rotary key
```

## The rotary part is served de-interleaved

The checkpoints store the rotary dimensions in interleaved pairs (`rope_interleave` is true), and
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

## The query scale carries YaRN's factor

The DeepSeek checkpoints extend the context with YaRN (`rope_scaling` factor 40, `mscale_all_dim`
1.0), and the attention multiplies the scores by `qk_head_dim ** -0.5` times 1.87
(`self_attn.scaling`, 0.1352), not by `qk_head_dim ** -0.5` alone; recompute `attention_scores`
from the queries and keys with `self_attn.scaling`. Moonlight has no `rope_scaling`, and its scale
is `qk_head_dim ** -0.5`.

## The mixture: sigmoid scores, a selection bias, renormalized weights

256 routed experts, 8 per token, and one shared expert every token runs through (64, 6 and two
on Moonlight). `router_logits` are the logits before the sigmoid, and before the selection bias
(`router.e_score_correction_bias`). The bias is added to the sigmoids only to choose: the experts
sit in `n_group` 8 groups, each group scored by the sum of its two best biased scores, and a token
takes its 8 experts from its best `topk_group` 4 groups. `expert_weights` are the chosen experts'
sigmoids without the bias, renormalized to sum to one (`norm_topk_prob`) and multiplied by
`routed_scaling_factor`, so they sum to 2.5 at every token (2.446 on Moonlight). A written
`router_logits` moves both the choice and the weights; a written bias moves only the choice.

```python
moe = model.layers[3].mlp
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

The shared expert runs after the routed ones. Ablating an expert is zeroing the weight of every
slot that chose it; the other slots keep their weights, so the token's weights then sum to less
than 2.5:

```python
with model.trace(prompt):
    moe.expert_weights = moe.expert_weights.masked_fill(moe.expert_indices == 3, 0)
    ablated = model.logits.save()
```

Blocks 0 to 2 have a dense MLP: `support()` reports every mixture value missing there.
"""
