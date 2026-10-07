"""Falcon: TII's Falcon-7B, 11B, 40B, 180B and Falcon-RW, the three block layouts of FalconForCausalLM."""

MODEL_TYPE = "falcon"
TITLE = "Falcon"
SUBTITLE = (
    "A parallel block whose one LayerNorm feeds both sublayers, with one key/value head on 7B; the block adds "
    "the attention into the MLP's output tensor in place, so mlp_output is a copy taken as the MLP returns, "
    "and the attention does its own arithmetic."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "tiiuae/falcon-7b"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "Rocketknight1/tiny-random-falcon-7b"
#: Three layouts: 7B and 11B (parallel, one input_layernorm), 40B and 180B (parallel, ln_attn and ln_mlp), Falcon-RW
#: (sequential, input_layernorm and post_attention_layernorm, ALiBi). BLOCK picks one from each config.
CHECKPOINTS = [
    "tiiuae/falcon-7b", "tiiuae/falcon-7b-instruct",
    "tiiuae/falcon-11B",
    "tiiuae/falcon-40b", "tiiuae/falcon-40b-instruct", "tiiuae/falcon-180B",
    "tiiuae/falcon-rw-1b", "tiiuae/falcon-rw-7b",
]

PALETTE = {"hue": 325}
VLLM = True
QUIRKS = ["parallel-blocks", "tuple-blocks", "own-attention-arithmetic", "fused-qkv", "layernorm", "alibi"]

INTERIOR = [
    "attention_queries", "attention_keys", "attention_values",
    "attention_scores", "attention_probabilities", "attention_head_outputs",
]
#: Both parallel layouts add the attention into the MLP's output, then the residual.
PARALLEL_NOTE = ("The block adds the attention into the MLP's output, then the residual: "
                 "(mlp_output + attention_output) + input is bit-exact; this order of the sum differs by "
                 "rounding in bfloat16.")

#: 7B and 11B: one input_layernorm whose output both sublayers read (drawn once per branch, one node), no
#: post-norms; the two contributions join the stream in one add.
SEVEN_B = {
    "topology": "parallel",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "pre_norm_note": "One LayerNorm, drawn on both branches: the attention and the MLP read the same output "
                             "tensor, so an in-place edit of self_attn.input reaches the MLP too.",
            "contribution": "attention_output",
            "interior": INTERIOR,
            "detail": "{num_heads} heads × {head_dim}, {num_kv_heads} kv",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "input_layernorm",
            "pre_norm_note": "One LayerNorm, drawn on both branches: the attention and the MLP read the same output "
                             "tensor, so an in-place edit of self_attn.input reaches the MLP too.",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, GELU",
        },
    ],
    "identity_note": PARALLEL_NOTE,
}

#: 40B and 180B (new_decoder_architecture, two norms in parallel): each sublayer reads its own LayerNorm of the
#: block input, ln_attn and ln_mlp, and the two contributions join the stream in one add.
FORTY_B = {
    "topology": "parallel",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "ln_attn",
            "pre_norm_note": "A LayerNorm of the block input, the attention's own; it has no standard name.",
            "contribution": "attention_output",
            "interior": INTERIOR,
            "detail": "{num_heads} heads × {head_dim}, {num_kv_heads} kv",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "ln_mlp",
            "pre_norm_note": "A LayerNorm of the block input, the MLP's own; it has no standard name.",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, GELU",
        },
    ],
    "identity_note": PARALLEL_NOTE,
}

#: Falcon-RW: sequential, input_layernorm before the attention and post_attention_layernorm before the MLP.
RW = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "contribution": "attention_output",
            "interior": INTERIOR,
            "detail": "{num_heads} heads × {head_dim}, ALiBi",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, GELU",
        },
    ],
}

#: What the visualization draws, by the checkpoint's config: Falcon-RW is the one sequential layout; a
#: new_decoder_architecture config with two norms in parallel is 40B's (11B sets num_ln_in_parallel_attn to 1).
BLOCK = [
    (lambda config: not config.parallel_attn, RW),
    (lambda config: config.new_decoder_architecture and config.num_ln_in_parallel_attn != 1, FORTY_B),
    (lambda config: True, SEVEN_B),
]

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "word_embeddings is a plain lookup with no scale and no position embedding (position enters in each "
             "attention, through rotary, or ALiBi on Falcon-RW), so token_embeddings equals layers[0].input.",
    "norm": "ln_f is a LayerNorm with a bias; project_on_vocab applies it, bias included.",
    "head": "lm_head has no bias; it is tied to word_embeddings on 7B, 40B and Falcon-RW, and has its own matrix on 11B. Nothing "
            "follows it: logits equals lm_head.output.",
}

NOTES = """
## One LayerNorm feeds both sublayers

```
h   = input_layernorm(x)
m   = mlp(h)
m  += self_attn(h)          # in place, into the MLP's output tensor
out = m + x
```

`input_layernorm` is the block's only norm, and its one output tensor is passed to the attention
and then to the MLP: `self_attn.input` and `mlp.input` are the same tensor. An in-place edit of
`self_attn.input` therefore reaches the MLP; an assignment replaces only the attention's
argument. Nothing inside a block connects the two sublayers: the first MLP that reads block `i`'s
attention is block `i + 1`'s. `ffn_hidden_size` is the MLP's width, which nnterp serves as
`intermediate_size`.

## `mlp_output` is a copy, and the live tensor is not the MLP's

The block adds the attention's output into the MLP's output tensor in place. Read raw, after the
block has run, `mlp.output` holds `mlp + attn`. `mlp_output` is a copy taken as the MLP returns,
so it holds the MLP's contribution alone, and a transform hands an edit of the copy back to the
model: an in-place edit and an assignment both land.

```python
with model.trace(prompt):
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    live = model.layers[1].mlp.output.save()

assert torch.equal(live, mlp + attn)            # the live tensor after the in-place add
```

The block's own order is `(mlp + attn) + x`, so that sum reproduces `layer_output` bit for bit;
`x + attn + mlp` differs from it by rounding in bfloat16.

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

assert torch.equal((mlp + attn) + x, out)
```

## One key/value head on 7B, eight broadcast on 11B

`query_key_value` is one `Linear` with no bias. On 7B (`multi_query`) its output is every head's
query, then one key, then one value: `view(batch, seq, num_heads + 2, head_dim)`, the last two
rows the key and the value. `attention_keys` and `attention_values` have one head,
`[batch, 1, seq, 64]`, and an edit to it reaches all 71 query heads. 11B groups its output by
key/value head, `[q × 4 | k | v]` for each of 8 groups, and broadcasts each key and value to its
four query heads before the rotary, so its `attention_keys` and `attention_values` are
`num_heads` (32) wide while `model.num_kv_heads` is 8; four heads hold each key, and an edit to
one of the four copies reaches only that query head.

```python
attn = model.layers[1].self_attn
with model.trace(prompt):
    h = model.layers[1].input_layernorm.output.save()
    v = attn.attention_values.save()

fused = h @ attn.query_key_value.weight.T
rows = fused.view(*fused.shape[:2], model.num_heads + 2, model.head_dim)    # the 7B layout
torch.testing.assert_close(rows[:, :, -1:].transpose(1, 2), v)
```

## Read the values before the queries and keys

The attention does its own arithmetic. The queries and keys are the two returns of the rotary
embedding, and the values are bound before it, so inside one trace read `attention_values`
first. Asking for the queries or keys first raises `OutOfOrderError` naming `value_layer_0`.

```python
attn = model.layers[1].self_attn
with model.trace(prompt):
    v = attn.attention_values.save()       # binds first
    q = attn.attention_queries.save()
    k = attn.attention_keys.save()
```

Rotary turns the whole head in `rotate_half` pairs, with `rope_theta` 10000 on 7B and 500042 on
11B. The pattern is the softmax's output (no dropout follows it), and the head outputs are the
`pattern @ values` product, heads first in the forward and served `[batch, seq, heads, head_dim]`.

## The interior needs an eager load

transformers' `FalconAttention` runs `sdpa` unless told otherwise, and then neither scores nor
pattern exist: the six interior values carry `⚠` and report `load with
attn_implementation='eager'`. `attention_output` and `mlp_output` do not depend on the
implementation.

```python
model = StandardizedTransformer("tiiuae/falcon-7b", attn_implementation="eager")
```

## Two more layouts load as FalconForCausalLM

- Falcon-40B and Falcon-180B (`new_decoder_architecture`, two norms in parallel) give each
  sublayer its own LayerNorm of the block input, `ln_attn` and `ln_mlp`; neither has a standard
  name. The fused projection and the broadcast keys and values are 11B's, with 16 query heads
  per key/value head on 40B.
- Falcon-RW (`tiiuae/falcon-rw-1b`, `falcon-rw-7b`) is sequential, with `input_layernorm` and
  `post_attention_layernorm`, biased projections, and ALiBi in place of rotary. Its fused
  projection is laid out by head, `[q_h | k_h | v_h]`. With ALiBi the read order is the forward's:
  queries, then keys, then values; values first raises `OutOfOrderError` naming
  `query_layer_0`.
- On an ALiBi checkpoint the eager attention adds the bias twice: transformers builds it into the
  attention mask and adds it again to the scores, so `attention_scores` is
  `q @ kᵀ / sqrt(head_dim) + 2 * slope_h * j / sqrt(head_dim)`. The default `sdpa` load adds it
  once. On `falcon-rw-1b` in float32 the two loads' logits differ by up to 1.54, so an eager
  trace there runs a different model from the default one.
"""
