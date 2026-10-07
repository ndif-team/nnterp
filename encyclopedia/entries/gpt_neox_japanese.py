"""GPT-NeoX-Japanese: ABEJA's Japanese 2.7B model, GPTNeoXJapaneseForCausalLM."""

MODEL_TYPE = "gpt_neox_japanese"
TITLE = "GPT-NeoX-Japanese"
SUBTITLE = (
    "GPT-NeoX's module names on a sequential block that returns a tuple: the attention does its own "
    "arithmetic on one fused per-head projection, and the last block's attention hands a separate bias "
    "to the block, which attention_output includes."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "abeja/gpt-neox-japanese-2.7b"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-GPTNeoXJapaneseForCausalLM"
CHECKPOINTS = ["abeja/gpt-neox-japanese-2.7b"]

#: GPT-NeoX's kin: GPT-NeoX hashes to 99 (GLM sits at 74 to 94, Phi at 106).
PALETTE = {"hue": 102}
VLLM = False
QUIRKS = ["tuple-blocks", "own-attention-arithmetic", "fused-qkv", "layernorm"]

#: What the visualization draws: a sequential block, input_layernorm before the attention and
#: post_attention_layernorm before the MLP; no post-norms.
BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "pre_norm_note": "A LayerNorm with a bias, of the block input.",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} heads × {head_dim}, rotary",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "A LayerNorm with a bias of the stream after the attention has been added: GPT-NeoX's "
                             "name, but this block is sequential, so the MLP reads the attention's contribution.",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
    ],
    "identity_note": "attention_output is the module's output plus, on the last block, the dense_bias the block "
                     "adds, so the plain sum is exact on every block.",
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "embed_in is a plain lookup with no scale and no position embedding (position enters through rotary in "
             "each attention), so token_embeddings equals layers[0].input.",
    "norm": "final_layer_norm is a LayerNorm with a bias; project_on_vocab applies it, bias included.",
    "head": "embed_out is lm_head: no bias, and its weight is embed_in's, the same tensor. Its output is logits, "
            "unchanged.",
}

NOTES = """
## The block is sequential, under GPT-NeoX's names

```
h   = x + (attention(input_layernorm(x)) + dense_bias)    # dense_bias: last block only
out = h + mlp(post_attention_layernorm(h))
```

The modules are `input_layernorm`, `attention` (`self_attn`), `post_attention_layernorm` and `mlp`,
as on GPT-NeoX, but `post_attention_layernorm` normalizes the stream after the attention has been
added, so each block's MLP reads its attention's contribution. The block returns
`(hidden_states, attn_weights)` on every call: `layer_output` is the first element.

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mid = model.layers[1].post_attention_layernorm.input.save()

torch.testing.assert_close(mid, x + attn)
```

## The last block's attention bias is part of attention_output

None of the projections has a bias. The last block's attention holds one parameter, `dense_bias`
(`hidden_size` wide), and returns it as the third element of its output,
`(attn_output, attn_weights, dense_bias)`; the block adds it to the attention's output before the
residual. `attention_output` is that sum on the last block and the module's own output on the
others, so `layers[i].input + attention_output + mlp_output == layer_output` holds on every block
(bit-identical on 2.7B in float16). An edit of `attention_output` on the last block, in place or
assigned, is carried back to the module's output minus the bias, and a read alone leaves the
forward bit-identical.

```python
last = model.layers[-1]
bias = last.self_attn._module.dense_bias

with model.trace(prompt):
    module = last.self_attn.output[0].save()
    attn = last.self_attn.attention_output.save()

assert torch.equal(attn, module + bias)
```

On 2.7B `dense_bias` has a norm of 5.4, against a median of 78 for the norm of the last attention's
own output rows on a sample prompt. Zeroing `self_attn.output[0]` on the last block leaves the bias
in the stream; zeroing `attention_output` removes both.

## Queries, keys and values come from one projection, laid out per head

`query_key_value` is a `Linear` from `hidden_size` to `3 * hidden_size` with no bias. Its output is
viewed as `[batch, seq, num_heads, 3 * head_dim]`: each head's slice holds its queries, then its
keys, then its values. The values are read as cut; the queries and keys are read after the rotary.

```python
W = model.layers[1].self_attn.query_key_value.weight
heads, d = model.num_heads, model.head_dim

with model.trace(prompt):
    h = model.layers[1].self_attn.input.save()
    v = model.layers[1].self_attn.attention_values.save()

qkv = (h @ W.T).unflatten(-1, (heads, 3 * d)).transpose(1, 2)
q_raw, k_raw, v_again = qkv.split(d, dim=-1)                 # before the rotary
torch.testing.assert_close(v_again, v)
```

Rotary turns the whole head (`rotary_pct` is 1.0) by `rotate_half`: dimension `i` turns with
`i + head_dim / 2`, at base 10000. Keys and values are `num_heads` wide, with no grouping.

## The interior needs no eager load

The attention has one implementation, its own `_attn(query, key, value, mask)`, whatever
`attn_implementation` says, so a default load serves all six interior values and they carry no
load condition. Its arguments are `attention_queries`, `attention_keys` and `attention_values`,
heads first. The scores are a `baddbmm` scaled by `1 / sqrt(head_dim)` (0.1118 at 2.7B's `head_dim` of 80),
then the mask is added, so the masked
entries hold the dtype's minimum. The pattern is read after `attention_dropout`, before its cast to
the values' dtype, and the first return of `_attn` is `attention_head_outputs`.

```python
with model.trace(prompt):
    q = model.layers[1].self_attn.attention_queries.save()
    k = model.layers[1].self_attn.attention_keys.save()
    scores = model.layers[1].self_attn.attention_scores.save()

n = scores.shape[-1]
causal = torch.ones(n, n, dtype=torch.bool, device=scores.device).tril()
again = q @ k.transpose(-1, -2) / model.head_dim ** 0.5
torch.testing.assert_close(again[..., causal], scores[..., causal])
```

## Loading: the config names no dtype

`abeja/gpt-neox-japanese-2.7b`'s config names no dtype, so a load without `dtype` is float32,
about 11 GB; `dtype=torch.float16` halves it. The tokenizer (`GPTNeoXJapaneseTokenizer`, 32000
tokens, with a separate emoji table) adds no BOS: position 0 is the prompt's first token, and
`<|startoftext|>` (31996) is there to prepend by hand.

## The first position is a sink with a huge norm

On 2.7B in float16, on a 28-token Japanese prompt, the stream at position 0 has a norm of 6,300 to
7,900 from block 3's output to block 24's, against medians of about 470 to 1,500 at the other
positions. Blocks 3 to 29 put 69% to 88% of their pattern (averaged over heads and the later
queries) on key 0. Leave position 0 out of norms you scale a steering vector by and of activation
statistics.

## The readout: tied weights and a LayerNorm with a bias

`logits` is `lm_head.output` with nothing applied, and `project_on_vocab` applied to the last
block's `layer_output` equals `logits` exactly (difference 0.0 on 2.7B). `lm_head` is the native
`embed_out`, with no bias, and its weight is `embed_tokens.weight`, the same tensor.
`final_layer_norm`'s bias adds `lm_head.weight @ norm.bias` to every position's logits whatever the
input (a standard deviation of 0.39 logits on 2.7B). The model is 32 blocks × 2560, 32 heads of 80,
an MLP four times the width (`intermediate_multiple_size`), and `gelu`.
"""
