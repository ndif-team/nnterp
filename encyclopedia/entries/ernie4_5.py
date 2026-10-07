"""ERNIE 4.5 dense: Llama's block and Llama's names, with rotary turning adjacent pairs of dimensions."""

MODEL_TYPE = "ernie4_5"
TITLE = "ERNIE 4.5"
SUBTITLE = (
    "Llama's pre-norm block whose rotary turns adjacent pairs of query and key dimensions, (2i, 2i + 1), "
    "and whose grouped query heads are together wider than the residual stream."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "baidu/ERNIE-4.5-0.3B-Base-PT"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-Ernie4_5ForCausalLM"
CHECKPOINTS = ["baidu/ERNIE-4.5-0.3B-Base-PT", "baidu/ERNIE-4.5-0.3B-PT"]

#: ERNIE lineage: ernie4_5_moe takes 185. ernie4_5's hash (11) sits between mamba2 (8) and mamba (17); 180 is in
#: the widest free gap, between cohere (176) and cohere2 (188).
PALETTE = {"hue": 180}
VLLM = False
QUIRKS = ["interleaved-rotary"]

#: Shapes, identities and the rotary layout ran on the pinned tiny checkpoint and on ERNIE-4.5-0.3B-Base-PT in
#: float32 on a GPU; the norm gains are 0.3B-Base's. Every snippet ran on both.

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
    ],
}

STRIP = {
    "embed": "A plain lookup: token_embeddings equals layers[0].input. The tokenizer prepends nothing; <s> (id 1) is "
             "its BOS, and a prompt starts with it only when it is written.",
    "norm": "A plain RMSNorm, eps 1e-5: the gain is norm.weight (-0.18 to 13.19, mean 6.29 on 0.3B-Base). "
            "project_on_vocab applies it.",
    "head": "lm_head shares its weight with embed_tokens (tie_word_embeddings is true on 0.3B). logits is lm_head.output.",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))
out = h + mlp(post_attention_layernorm(h))
```

Llama's pre-norm block and Llama's names. Nothing norms a sublayer's output, so
`attention_output` is `o_proj`'s output and `mlp_output` the MLP's, and the identity is the
plain sum:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

No projection has a bias (`use_bias` is false).

## The heads are wider than the stream

0.3B has `hidden_size` 1024 and 16 query heads of `head_dim` 128, so `q_proj` maps 1024 to 2048 and
`o_proj` 2048 back to 1024. The 16 query heads share 2 key/value heads, 8 to each: an edit to key
head 0 at `attention_keys` reaches query heads 0 to 7. The attention interior needs
`attn_implementation="eager"`.

## Rotary turns adjacent pairs

The rotary embedding rotates dimensions (2i, 2i + 1) of each query and key head together, at
frequency `rope_theta ** (-2i / head_dim)` (`rope_theta` 500000), where Llama's rotates i with
i + 64. The served queries and keys keep the projection's order: at position 0 `attention_queries`
is `q_proj`'s output split into heads, and at position 1 dimensions 0 and 1 are that pair turned
by one radian:

```python
attn = model.layers[1].self_attn
with model.trace(prompt):
    proj = attn.q_proj.output.save()
with model.trace(prompt):
    q = attn.attention_queries.save()          # [batch, heads, seq, head_dim]

p = proj.view(1, -1, model.num_heads, model.head_dim)
torch.testing.assert_close(q[0, :, 0], p[0, 0])
c, s = math.cos(1.0), math.sin(1.0)
a, b = p[0, 1, 0, 0], p[0, 1, 0, 1]
torch.testing.assert_close(q[0, 0, 1, :2], torch.stack([a * c - b * s, b * c + a * s]))
```

A probe or a patch that splits a head into a rotating half and its partner, as a rotate-half
layout would, pairs the wrong dimensions here.

## The readout

`logits` is `lm_head.output`, with no cap or scale, and `project_on_vocab` on the last block's
`layer_output` equals `logits`. On 0.3B `lm_head` is `embed_tokens`'s matrix
(`tie_word_embeddings`).
"""
