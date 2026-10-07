"""BLOOM: BigScience's BLOOM and BLOOMZ, every checkpoint that loads as BloomForCausalLM."""

MODEL_TYPE = "bloom"
TITLE = "BLOOM / BLOOMZ"
SUBTITLE = (
    "Each sublayer takes the residual as an argument and adds it inside the module, so the modules return "
    "stream states and the contributions are the tensors before that add; ALiBi biases on the scores stand in "
    "for rotary, and a LayerNorm sits between the embedding and block 0."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "bigscience/bloom-560m"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-BloomForCausalLM"
CHECKPOINTS = [
    "bigscience/bloom-560m", "bigscience/bloom-1b1", "bigscience/bloom-1b7",
    "bigscience/bloom-3b", "bigscience/bloom-7b1", "bigscience/bloom",
    "bigscience/bloomz-560m", "bigscience/bloomz-1b1", "bigscience/bloomz-1b7",
    "bigscience/bloomz-3b", "bigscience/bloomz-7b1", "bigscience/bloomz-7b1-mt",
    "bigscience/bloomz", "bigscience/bloomz-mt",
]

PALETTE = {"hue": 242}
VLLM = True
QUIRKS = ["residual-inside-module", "tuple-blocks", "own-attention-arithmetic", "fused-qkv", "qkv-bias", "layernorm"]

#: What the visualization draws: a sequential block with two LayerNorm pre-norms; each module adds
#: the residual itself, and its contribution is the tensor entering that add.
BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "pre_norm_note": "A LayerNorm with a bias. Its output is the attention's input; the block also passes "
                             "the attention the raw stream, which the module adds to its own output.",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} heads × {head_dim}, ALiBi",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "It normalizes self_attn.output[0], the stream after the attention's add, which the "
                             "attention module returns. The MLP takes that stream too and adds it to its own output.",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, tanh GELU",
        },
    ],
    "identity_note": "Exact in every dtype: the attention module returns input + attention_output, and the MLP "
                     "module adds mlp_output to that, the same order as the sum.",
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "word_embeddings is a plain lookup, and word_embeddings_layernorm, a LayerNorm with no standard name, "
             "normalizes it before block 0: token_embeddings is the lookup, layers[0].input its normalization. "
             "No position embedding: position enters as ALiBi biases on each block's attention scores.",
    "norm": "ln_f is a LayerNorm with a bias; project_on_vocab applies it, bias included.",
    "head": "lm_head is tied to word_embeddings and has no bias. Nothing follows it: logits equals lm_head.output.",
}

NOTES = """
## Each module adds the residual itself

```
h   = self_attn(input_layernorm(x), residual=x)            # returns x + attention_output
out = mlp(post_attention_layernorm(h), residual=h)         # returns h + mlp_output
```

The block hands each sublayer the stream as an argument, and the module ends in `dropout_add`,
which adds it. So `self_attn.output[0]` is the stream after the attention's add, and
`mlp.output` is the block's output, equal to `layer_output`. `attention_output` and
`mlp_output` are the first argument of each `dropout_add`: the attention's `dense` output and
the MLP's `dense_4h_to_h` output, each with its bias. The sum is exact in every dtype.

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()
with model.trace(prompt):
    raw = model.layers[1].self_attn.output.save()

assert torch.equal(raw[0], x + attn)          # the module returns the stream
assert torch.equal(x + attn + mlp, out)
```

## Ablate the contribution, not the module output

Zeroing `attention_output` removes the attention's contribution and leaves the stream: the MLP's
norm then reads `layers[i].input` unchanged. Zeroing `self_attn.output[0]` zeroes the stream
itself, and the MLP and every later block see zeros. The same holds for `mlp_output` against
`mlp.output`. Steering, patching and attribution on a contribution go through the standard
names for the same reason.

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    model.layers[1].self_attn.attention_output[:] = 0
    h = model.layers[1].post_attention_layernorm.input.save()

assert torch.equal(h, x)
```

The contribution is an operation inside the module, so in one trace read it before the module's
own `.output`. Asking for `self_attn.output` first and `attention_output` after raises
`OutOfOrderError` naming `self_attention.source.dropout_add_0.input`.

## Queries, keys and values come out of one projection, head by head

`query_key_value` is one `Linear` with a bias, `hidden_size` to `3 * hidden_size`. Its output is
laid out by head: each head's query, key and value sit side by side, `[q_h | k_h | v_h]`, so it
splits as `view(batch, seq, heads, 3, head_dim)`, not into thirds. The weight's rows follow the
same order. The three values are views of that output; in-place edits of them land.

```python
attn = model.layers[1].self_attn
with model.trace(prompt):
    h = model.layers[1].input_layernorm.output.save()
    q = attn.attention_queries.save()

fused = torch.nn.functional.linear(h, attn.query_key_value.weight, attn.query_key_value.bias)
per_head = fused.view(*fused.shape[:2], model.num_heads, 3, model.head_dim)
torch.testing.assert_close(per_head[..., 0, :].transpose(1, 2), q)
W = attn.query_key_value.weight.view(model.num_heads, 3, model.head_dim, -1)
W_q = W[:, 0]                                  # [heads, head_dim, hidden]
```

## ALiBi biases replace rotary

There is no rotary embedding and no position embedding: queries and keys carry no position.
The same token at two positions gives the same `attention_queries` and `attention_keys` at
block 0. Position enters only through a bias on the scores: `attention_scores` is
`q @ kᵀ / sqrt(head_dim)` plus `slope_h * j` for key position `j`, then the causal mask. The bias
grows with the key's position and is the same on every query row, so after the softmax it acts
as a penalty of `slope_h` per token of distance. With a power-of-two head count head `h` has the
slope `2 ** (-8 * (h + 1) / heads)`: on 560m, with 16 heads, 0.71 for head 0 down to 0.0039 for
head 15. An edit to the queries or keys leaves the bias as it was. On a float32 load the scores
recompute to rounding:

```python
from transformers.models.bloom.modeling_bloom import build_alibi_tensor

attn = model.layers[1].self_attn
with model.trace(prompt):
    q = attn.attention_queries.save()
    k = attn.attention_keys.save()
    scores = attn.attention_scores.save()

seq = q.shape[2]
alibi = build_alibi_tensor(torch.ones(1, seq), model.num_heads, q.dtype).view(1, model.num_heads, 1, seq)
again = q @ k.transpose(-1, -2) / model.head_dim ** 0.5 + alibi
causal = torch.ones(seq, seq, dtype=torch.bool).tril()
torch.testing.assert_close(again[..., causal], scores[..., causal])
```

## The attention needs no flag

transformers has one attention for BLOOM, its own arithmetic, so a load without
`attn_implementation` runs it and all six interior values are served: `support()` reports none
missing. The scores are read where the softmax takes them; the softmax runs in float32 and the
pattern is cast back to the model's dtype. The head outputs are the `bmm` of the pattern and the
values, `[batch * heads, seq, head_dim]` in the forward, served `[batch, seq, heads, head_dim]`.
`bigscience/bloom-560m` stores float16 weights, and a load without `dtype` is float16.

## A LayerNorm sits between the embedding and block 0

`token_embeddings` is `word_embeddings`' output, and `layers[0].input` is that tensor through
`word_embeddings_layernorm`, so the two differ. The norm has no standard name; it is
`model.transformer.word_embeddings_layernorm`. On 560m the embedding rows are small (mean
norm 0.37) and the first block's input has a norm of about 10 per token. An edit at
`token_embeddings` passes through that LayerNorm, which rescales it with the rest of the row;
edit `layers[0].input` to set what block 0 reads.

```python
with model.trace(prompt):
    emb = model.token_embeddings.save()
    first = model.layers[0].input.save()

assert torch.equal(model.transformer.word_embeddings_layernorm(emb), first)
```

## The readout

`lm_head` is `word_embeddings`' matrix (tied) with no bias, and `logits` equals
`lm_head.output`. `ln_f` is a LayerNorm with a bias; `project_on_vocab` on the last block's
`layer_output` equals `logits`. The tokenizer has 250680 entries and the matrix 250880 rows; the
last 200 ids are padding. The tokenizer adds no BOS: position 0 is the prompt's first token.
"""
