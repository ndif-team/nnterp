"""Seed-OSS: ByteDance Seed's 36B dense models, which load as SeedOssForCausalLM."""

MODEL_TYPE = "seed_oss"
TITLE = "Seed-OSS"
SUBTITLE = (
    "Llama's block with a bias on the query, key and value projections and an attention twice as wide "
    "as the stream: 80 query heads of 128 over 8 key/value heads, in a 5120-wide stream."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "ByteDance-Seed/Seed-OSS-36B-Base"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-SeedOssForCausalLM"
CHECKPOINTS = [
    "ByteDance-Seed/Seed-OSS-36B-Base",
    "ByteDance-Seed/Seed-OSS-36B-Base-woSyn",
    "ByteDance-Seed/Seed-OSS-36B-Instruct",
]

PALETTE = {"hue": 21}
VLLM = False
QUIRKS = ["qkv-bias"]

#: What the visualization draws: Llama's sequential block, one RMSNorm before each sublayer.
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
            "detail": "{num_heads} heads, {num_kv_heads} kv; q, k, v biased",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "Named for its place on the stream, after the attention's add: it is the MLP's input norm.",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "A plain lookup: no scale and no position embedding, so token_embeddings equals layers[0].input. "
             "The tokenizer prepends no BOS.",
    "norm": "RMSNorm whose gain is norm.weight as stored.",
    "head": "lm_head has its own weight, not tied to embed_tokens, and no bias. Nothing follows it: logits "
            "equals lm_head.output.",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))
out = h + mlp(post_attention_layernorm(h))
```

Llama's block and Llama's names: two RMSNorms, each before its sublayer, none after. The attention
and the MLP each end in a residual dropout inside the module (`residual_dropout`, `0.1` in every
config), after `o_proj` and after `down_proj`. A loaded model is in eval mode, where the dropout is
the identity.

## The contributions are the modules' outputs

`attention_output` is `self_attn.output[0]` and `mlp_output` is `mlp.output`, added to the stream
with nothing in between, and the identity holds exactly in float32 on the pinned checkpoint:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.equal(x + attn + mlp, out)   # True
```

Under `model.train()` (fine-tuning an adapter through the model) the residual dropout is live. The
contributions are then read after it, so `attention_output` differs from `o_proj.output`, and
the identity still holds: on the pinned checkpoint with `residual_dropout=0.5`, about half of
`attention_output`'s entries are zero and `x + attn + mlp == out` stays exact.

## The attention is twice as wide as the stream

The 36B checkpoints have `hidden_size` 5120 and 80 query heads of `head_dim` 128, so `q_proj` maps
5120 to 10240 and `o_proj` maps 10240 back to 5120. `attention_head_outputs` is
`[batch, seq, 80, 128]`, and the concatenated heads are twice the stream's width: a head's output is
not a slice of a stream vector until `o_proj` has mapped it.

There are 8 key/value heads, so `attention_keys` and `attention_values` are `[batch, 8, seq, 128]`
and query head `h` reads key/value head `h // 10`: an edit to key/value head `j` reaches query
heads `10j` to `10j + 9`.

```python
attn = model.layers[1].self_attn
with model.trace(prompt):
    k = attn.attention_keys.save()            # [batch, num_kv_heads, seq, head_dim]
    heads = attn.attention_head_outputs.save()  # [batch, seq, num_heads, head_dim]
```

## Queries, keys and values carry a bias

`q_proj`, `k_proj` and `v_proj` add a bias (`attention_bias`); `o_proj` does not
(`attention_out_bias` is `false`), nor do the three MLP projections (`mlp_bias`). The bias is added
before the rotary embedding, so it is rotated with the rest of the query and key: with the
attention's input zeroed, `attention_keys` is the rotated `k_proj.bias`, equal to the bias at
position 0 and turned at every other position (checked on the pinned checkpoint with its zero
bias filled with ones).

```python
with model.trace(prompt):
    model.layers[1].self_attn.input[:] = 0
    k = model.layers[1].self_attn.attention_keys.save()   # rope(k_proj.bias), not zero
```

To remove the attention's effect, ablate `attention_output` itself.

## The default load runs the same attention as eager

A default load runs `sdpa`, and the six attention interior values report unavailable;
`attn_implementation="eager"` serves them. There is no softcap, window or sink, so the two compute
the same function. The query scale is `head_dim ** -0.5`. The rotary is the plain one with
`rope_theta` 10000000, over a `max_position_embeddings` of 524288.

## The readout and the tokenizer

`model.logits` is `model.lm_head.output`, so `project_on_vocab` is `lm_head(norm(hidden))`, and
`lm_head` has its own weight (`tie_word_embeddings` is `false`). The tokenizer prepends no BOS
(`<seed:bos>`, id 0, is not added): position 0 is the text's first token. It has 155121 tokens
against a `vocab_size` of 155136; the 15 ids past the tokenizer get logits but decode to nothing.

## Three checkpoints, one shape

`Seed-OSS-36B-Base`, `Seed-OSS-36B-Base-woSyn` and `Seed-OSS-36B-Instruct` have the same config:
64 blocks, the same widths, heads and rotary. `-woSyn` is the base model pretrained without
synthetic instruction data, released for research that needs a base model untouched by it.
"""
