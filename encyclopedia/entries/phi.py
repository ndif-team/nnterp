"""Phi 1 / 1.5 / 2: Microsoft's small models that load as PhiForCausalLM."""

MODEL_TYPE = "phi"
TITLE = "Phi-1 / Phi-1.5 / Phi-2"
SUBTITLE = (
    "A parallel block: one LayerNorm feeds attention and MLP and the block adds both at once; the query, key, "
    "value and output projections carry a bias, rotary turns 32 dimensions of each head, and lm_head has a bias."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "microsoft/phi-2"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-PhiForCausalLM"
CHECKPOINTS = ["microsoft/phi-1", "microsoft/phi-1_5", "microsoft/phi-2"]

#: The Phi lineage (phi, phi3, phimoe) sits in the free band between GPT-NeoX's hashed 99 and Gemma-4 unified's 127.
PALETTE = {"hue": 106}
VLLM = True
QUIRKS = ["parallel-blocks", "partial-rotary", "qkv-bias", "layernorm"]

#: Every real value in the notes was measured on microsoft/phi-2 (bfloat16 on CPU unless the note says float16,
#: which ran on a GPU) and, where named, microsoft/phi-1_5; the snippets also ran on the pinned tiny checkpoint.

#: What the visualization draws: one input_layernorm feeds both sublayers; the schema draws a pre-norm per
#: sublayer, so it appears twice, as one node.
BLOCK = {
    "topology": "parallel",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "pre_norm_note": "One module, drawn on both rows: input_layernorm's output is what the attention and the "
                             "MLP both read, so self_attn.input equals mlp.input. The block has no "
                             "post_attention_layernorm.",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} heads × {head_dim}, partial rotary",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "input_layernorm",
            "pre_norm_note": "One module, drawn on both rows: input_layernorm's output is what the attention and the "
                             "MLP both read, so self_attn.input equals mlp.input. The block has no "
                             "post_attention_layernorm.",
            "contribution": "mlp_output",
            "detail": "fc1 {hidden_size} → {intermediate_size}, {hidden_act}, fc2 → {hidden_size}",
        },
    ],
    "identity_note": "The block adds in its own order, attention_output + mlp_output + layers[i].input, which equals "
                     "layer_output bit for bit; the sum in the identity's order differs from it by rounding.",
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "embed_tokens is a plain lookup: no scale and no position embedding (position enters through rotary in "
             "each attention), so token_embeddings equals layers[0].input. The tokenizer adds no BOS.",
    "norm": "final_layernorm is a LayerNorm with a bias; project_on_vocab applies it, bias included.",
    "head": "lm_head has its own weight, not tied to embed_tokens (tie_word_embeddings is false), and a bias. Nothing "
            "follows it: logits equals lm_head.output.",
}

NOTES = """
## The block is parallel, with one LayerNorm

```
h   = input_layernorm(x)
out = self_attn(h) + mlp(h) + x
```

`input_layernorm` is the block's only norm. Its output is both `self_attn.input` and `mlp.input`,
the same tensor, and the block has no `post_attention_layernorm`. `attention_output` is `dense`'s
output and `mlp_output` is `fc2`'s, and nothing normalizes either before the add. Nothing connects
the two sublayers of one block: zeroing `attention_output` leaves that block's `mlp.input`
bit-identical, so block `i`'s attention first reaches an MLP at block `i + 1`.

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp_in = model.layers[1].mlp.input.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

assert torch.equal(attn + mlp + x, out)   # the block's own order, bit for bit

with model.trace(prompt):
    model.layers[1].self_attn.attention_output[:] = 0
    mlp_in_ablated = model.layers[1].mlp.input.save()

assert torch.equal(mlp_in, mlp_in_ablated)
```

The block sums the attention and the MLP first and adds the residual last. In the identity's
order, `x + attn + mlp`, the sum differs from `layer_output` by rounding: by up to 7e-9 on the
pinned checkpoint in float32, and by up to 2.0 over phi-2's blocks in bfloat16, whose stream
entries reach 800, enough to fail `torch.testing.assert_close` at its bfloat16 tolerance.
The block returns a bare tensor, so `model.layers[i].output` is `layer_output`.
`resid_dropout` wraps both contributions and `embed_dropout` the embeddings (`resid_pdrop` is
0.1 on phi-2); both are inert in eval.

## Every projection in the attention has a bias

`self_attn.q_proj`, `k_proj`, `v_proj` and `dense` are separate `Linear`s, each with a bias:
`dense` is the attention's output projection, not a layer of the MLP. The MLP is `fc1`, the
activation (`gelu_new` on all three checkpoints) and `fc2`, both with a bias. Keys and values have
`num_heads` heads: phi-1 and phi-1.5 leave `num_key_value_heads` unset, which transformers fills
with `num_attention_heads`, and phi-2 sets it to 32. `qk_layernorm` is false on all three, so
nothing normalizes the queries and keys.

## Rotary turns 32 dimensions of each head

`partial_rotary_factor` is 0.5 on phi-1 and phi-1.5, 0.5 of 64-dimensional heads, and 0.4 on phi-2,
0.4 of 80: 32 dimensions on every checkpoint. The rotation is `rotate_half` on those 32, pairing
dimension `i` with `i + 16`; the other dimensions of every query and key are the projection's
output unchanged and carry no position. `attention_queries` and `attention_keys` are read after the
rotary; for the queries before it, read the projection in the same trace:

```python
with model.trace(prompt):
    q_raw = model.layers[1].self_attn.source.self_q_proj_0.output.save()
    q = model.layers[1].self_attn.attention_queries.save()

b, s, _ = q_raw.shape
q_pre = q_raw.view(b, s, model.num_heads, model.head_dim).transpose(1, 2)
rot = int(model.head_dim * model.config.rope_parameters["partial_rotary_factor"])   # 32
assert torch.equal(q_pre[..., rot:], q[..., rot:])   # past rot: no rotation
```

The scale is `head_dim ** -0.5`, and `softmax(attention_scores)` equals
`attention_probabilities`.

## phi-2 in float16 under eager attention returns NaN

phi-1.5 and phi-2 are stored in float16 (`torch_dtype`), phi-1 in float32, and a load without
`dtype` keeps it. The eager path multiplies queries by keys and scales the product after, and on
phi-2 the unscaled product passes float16's largest value, 65504: on
`"The Eiffel Tower is in the city of"` it reaches 1.1e6 at block 29 and stays above 65504 at blocks
30 and 31. Under `attn_implementation="eager"` in float16, block 29's `attention_output` is not
finite and every logit is NaN. The default `sdpa` load in float16 is finite (p(` Paris`) = 0.94),
and so is eager in bfloat16 (0.92). Load the attention interior in bfloat16 or float32:

```python
model = StandardizedTransformer(
    "microsoft/phi-2", dispatch=True, attn_implementation="eager", dtype=torch.bfloat16
)
```

On phi-1.5 the same prompt's largest product is 8.7e3, inside float16's range. With the dtype
set, the eager and `sdpa` loads compute the same attention: no softcap, sink or window.

## Position 0 carries one very large entry

The stream at the first position has a single dimension far larger than anything else: at block 16
of phi-2 on the sample prompt, dimension 743 of position 0 is 792, and no entry at any other
position exceeds 17. phi-1.5 shows the same at block 12, 684 at dimension 725 against 19
elsewhere. The tokenizer adds no BOS, so position 0 is the prompt's own first token. A norm or a
mean of the stream taken over positions is dominated by it; leave position 0 out, or read it on its
own.

## The head has a bias, and a constant logit vector

`logits` equals `lm_head.output`, and `project_on_vocab` applied to the last block's
`layer_output` equals `logits` exactly, `final_layernorm`'s bias and `lm_head`'s included. A logit
lens therefore adds the same vector at every block, whatever the input:

```python
bias_logits = model.lm_head.weight @ model.norm.bias + model.lm_head.bias
```

On phi-2 its standard deviation is 0.33 logits, against 2.3 for a whole lens readout at block 16
on the sample prompt (`lm_head.bias` alone: 0.04), and its top tokens are `\\n`, `-`, `.`, `,` and
`\\n\\n`. `embed_tokens` and `lm_head` are separate matrices. The vocabulary is padded: 51200 rows
against 50295 tokenizer tokens. The 905 extra rows are not zero; on the sample prompt their logits
are at most -1.1 and their total probability 1.2e-6.

## The tokenizer is CodeGen's

The three checkpoints ship CodeGen's `-mono` tokenizer: GPT-2's 50257 tokens plus one token for
each run of 2 to 31 spaces and of 2 to 9 tabs. A run of two or more spaces is one token, and the
word after it loses its leading space:

```python
model.tokenizer("    return x").input_ids   # [50284, 7783, 2124]: '    ', 'return', ' x'
```

`<|endoftext|>` (50256) is its `bos_token` and `eos_token`, and the tokenizer adds neither.

## Three checkpoints

The sizes, from the configs:

- phi-1 (1.3B): 24 blocks, hidden 2048, 32 heads × 64, `float32` files
- phi-1.5 (1.3B): 24 blocks, hidden 2048, 32 heads × 64, `float16` files
- phi-2 (2.7B): 32 blocks, hidden 2560, 32 heads × 80, `float16` files

All three set a context of 2048 tokens and untied embeddings.
"""
