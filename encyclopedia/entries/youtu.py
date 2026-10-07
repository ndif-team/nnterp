"""Youtu-LLM: DeepSeek-V2's latent attention on Llama's dense block, with tied embeddings."""

MODEL_TYPE = "youtu"
TITLE = "Youtu-LLM"
SUBTITLE = (
    "Llama's dense block with DeepSeek-V2's latent attention: queries and keys wider than values, "
    "and every head's key and value expanded from one latent per token."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "tencent/Youtu-LLM-2B-Base"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-YoutuForCausalLM"
#: Every public id whose config is youtu (Youtu-LLM-2B-GGUF has no config.json).
CHECKPOINTS = ["tencent/Youtu-LLM-2B-Base", "tencent/Youtu-LLM-2B"]

#: DeepSeek lineage: deepseek_v2 357, deepseek_v3 2; Youtu's latent attention sits beside them.
PALETTE = {"hue": 9}
VLLM = False
QUIRKS = ["latent-attention", "interleaved-rotary"]

#: Shapes, identities and the rotary order were run on the pinned tiny checkpoint (queries and keys 64 wide,
#: values 32). Sizes, the scale, tied weights and the tokenizer are the 2B checkpoints', read off their
#: configs, meta builds and tokenizers; no real weights were run.

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
    ],
}

STRIP = {
    "embed": "A plain lookup: token_embeddings equals layers[0].input. The tokenizer prepends nothing.",
    "head": "lm_head is embed_tokens' weight (tie_word_embeddings is true); logits is lm_head.output.",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))      # multi-head latent attention
out = h + mlp(post_attention_layernorm(h))   # a dense gated MLP on every block
```

Llama's pre-norm block and Llama's names. The block adds each sublayer's output to the stream and
returns a tensor, so the identity is the plain sum:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

## Latent attention: queries and keys are wider than values

`attention_queries` and `attention_keys` are `qk_head_dim` wide (`qk_nope_head_dim +
qk_rope_head_dim`, 128 + 64 = 192), `attention_values` and `attention_head_outputs` `head_dim`
(`v_head_dim`, 128). The config's `head_dim` is set to `qk_rope_head_dim` (64), which no served
value has. Queries go through a compressed rank (`q_lora_rank` 1536: `q_a_proj`, `q_a_layernorm`,
`q_b_proj`; `self_attn.q_proj` is `None`). Keys and values are expanded from one latent per token
(`kv_lora_rank` 512) for every head, so `attention_keys` has `num_heads` (16) heads, and the last
64 dimensions of every head's key are one shared rotary key. The scores are scaled by
`qk_head_dim ** -0.5` (`self_attn.scaling`, 0.0722). The attention interior needs
`attn_implementation="eager"`.

```python
attn = model.layers[1].self_attn
with model.trace(prompt):
    q = attn.attention_queries.save()          # [1, 16, seq, 192]
    k = attn.attention_keys.save()             # [1, 16, seq, 192]
    v = attn.attention_values.save()           # [1, 16, seq, 128]

assert torch.equal(k[:, 0, :, -64:], k[:, 1, :, -64:])   # one rotary key
```

An edit to one head's last 64 key dimensions reaches that head only: the expansion has already
copied the rotary key into every head when `attention_keys` is read.

## The rotary part is served de-interleaved

`rope_interleave` is true: the attention rotates pair (2i, 2i + 1) of the projection into
dimensions i and i + 32 of the rotary part. `attention_queries[..., -64:]` and
`attention_keys[..., -64:]` are in that rotate-half order, not the order of `q_b_proj`'s output:
at position 0, where rotary is the identity, the served rotary part is the projection's even
dimensions followed by its odd ones.

```python
with model.trace(prompt):
    proj = attn.q_b_proj.output.save()
with model.trace(prompt):
    q = attn.attention_queries.save()

rope = proj.view(1, -1, model.num_heads, model.qk_head_dim)[0, 0, 0, -64:]
torch.testing.assert_close(q[0, 0, 0, -64:], torch.cat([rope[0::2], rope[1::2]]))
```

## The readout is tied

`lm_head.weight` is `embed_tokens.weight`, so a direction read off the unembedding is also the
embedding's: a write to one weight changes both. `logits` is `lm_head.output`, with no softcap
or scale, and `project_on_vocab` applies `norm` and then the tied head.
"""
