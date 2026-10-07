"""Cohere: Command R and Command R+ (CohereForCausalLM)."""

MODEL_TYPE = "cohere"
TITLE = "Command R / Command R+"
SUBTITLE = (
    "A parallel block whose one LayerNorm, without a bias, feeds both the attention and the MLP; rotary turns "
    "adjacent pairs, Command R+ norms queries and keys per head, and the logits are the head's output times logit_scale."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
#: Every CohereLabs repository is gated; this is an ungated copy of c4ai-command-r-08-2024 (its card names that base).
REFERENCE = "unsloth/c4ai-command-r-08-2024"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "trl-internal-testing/tiny-CohereForCausalLM"
CHECKPOINTS = [
    "CohereLabs/c4ai-command-r-v01",
    "CohereLabs/c4ai-command-r-plus",
    "CohereLabs/c4ai-command-r-08-2024",
    "CohereLabs/c4ai-command-r-plus-08-2024",
    # Ungated copies of the 08-2024 releases.
    "unsloth/c4ai-command-r-08-2024",
    "unsloth/c4ai-command-r-plus-08-2024",
]

#: Set by hues.py (lineage: Cohere).
PALETTE = {"hue": 174}
VLLM = True
QUIRKS = ["parallel-blocks", "layernorm", "interleaved-rotary", "qk-norm", "scaled-logits"]

#: What the visualization draws: one input_layernorm whose output both sublayers read (drawn once per branch,
#: one node), no post-norms; the two contributions join the stream in one add.
BLOCK = {
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
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} query, {num_kv_heads} key/value heads",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "input_layernorm",
            "pre_norm_note": "One LayerNorm, drawn on both branches: the attention and the MLP read the same output "
                             "tensor, so an in-place edit of self_attn.input reaches the MLP too.",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
    ],
    "identity_note": "Exact in the block's own order, x + attn + mlp; another order differs by rounding.",
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "A plain lookup with no scale and no position embedding, so token_embeddings equals layers[0].input. "
             "Positions enter as a rotation of queries and keys inside each attention.",
    "norm": "CohereLayerNorm: subtracts the mean and multiplies by a weight, with no bias, computed in float32.",
    "head": "lm_head shares its weight with embed_tokens (tie_word_embeddings).",
    "logits": "logits = lm_head.output × logit_scale (0.0625 on Command R 08-2024). project_on_vocab multiplies too.",
}

NOTES = """
## One LayerNorm feeds both sublayers

```
h   = input_layernorm(x)
out = x + self_attn(h) + mlp(h)
```

`input_layernorm` is the block's only norm: there is no `post_attention_layernorm`. Its one output
tensor is passed to the attention and then to the MLP, so `self_attn.input` and `mlp.input` are the
same tensor, not two equal ones. An in-place edit of `self_attn.input` therefore also changes what
the MLP reads; an assignment replaces only the attention's argument. To change what both read,
assign `input_layernorm.output`.

```python
with model.trace(prompt):
    model.layers[1].self_attn.input[:, -1] = 0      # in place
    mlp_in = model.layers[1].mlp.input.save()       # its last row is zero too
```

## The contributions are the two modules' outputs

`attention_output` is `o_proj`'s output and `mlp_output` is `down_proj`'s; neither has a bias. The
block computes `residual + attn + mlp`, so that sum reproduces `layer_output` bit for bit, in
float32 and in bfloat16. Summed in another order (`attn + mlp + x`) it differs by rounding. Nothing
inside a block connects the two sublayers: zeroing `attention_output` leaves that block's
`mlp.input` bit-identical, and the first MLP that reads block `i`'s attention is block `i + 1`'s.

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

assert torch.equal(x + attn + mlp, out)           # the block's own order
```

## The norms are LayerNorms with a weight and no bias

`CohereLayerNorm` casts to float32, subtracts the mean, divides by the standard deviation
(`layer_norm_eps`, 1e-5) and multiplies by `weight`; there is no bias term, and the result is cast
back to the model's dtype. `input_layernorm` and the final `norm` are this class. Folding a norm
into the next matrix takes `weight` alone, and the mean subtraction stays as a projection that
removes the all-ones direction.

## Rotary turns adjacent pairs

Cohere's rotary pairs dimension `2i` with `2i + 1` (its `rotate_half` takes the even and odd
dimensions, and each frequency in the cos/sin table appears twice in a row), where Llama pairs `i` with
`i + head_dim / 2`. It turns the whole head. `attention_queries` and `attention_keys` are read
after the rotation; the rotation itself is computed in float32 and cast back. A comparison with a
`rotate_half` family's queries, or rotary code written for one, needs the head reordered, evens then
odds. The base (`rope_theta`) is 8,000,000 on Command R v01 and 4,000,000 on Command R 08-2024.

```python
hd = model.head_dim
perm = torch.cat([torch.arange(0, hd, 2), torch.arange(1, hd, 2)])

with model.trace(prompt):
    q = model.layers[1].self_attn.attention_queries.save()

q_half = q[..., perm]          # in rotate_half's order
```

## Command R+ norms queries and keys per head

Where `use_qk_norm` is true (Command R+ and Command R+ 08-2024), the attention has `q_norm` and
`k_norm`, `CohereLayerNorm`s over each head's `head_dim` vector whose weight is
`[heads, head_dim]`: every head has its own gain. They act on the projections before the rotary,
on `[batch, seq, heads, head_dim]` tensors, and `attention_queries` and `attention_keys` are read
after both. Command R and Command R 08-2024 have no `q_norm` or `k_norm`.

```python
with model.trace(prompt):
    normed = model.layers[1].self_attn.q_norm.output.save()
    # [batch, seq, heads, head_dim], before the rotary
```

## Attention

Command R v01 has 64 query heads and 64 key/value heads (no grouping); Command R 08-2024 has 64
and 8, and Command R+ 96 and 8, so an edit to a key or value head there reaches 8 or 12 query heads.
`attention_keys` and `attention_values` are served before `repeat_kv`. The query scale is
`head_dim ** -0.5` (128 on every size). The attention runs the shared attention interface, so the
values read inside it (marked `⚠`) need `attn_implementation="eager"`.

## The logits are the head's output times `logit_scale`

`lm_head` is the embedding matrix (`tie_word_embeddings`), and the model multiplies its output by
`config.logit_scale`: 0.0625 on Command R v01 and 08-2024, 0.8333 on Command R+ 08-2024. `logits`
is the scaled tensor and `lm_head.output` the unscaled one. The family's `project_on_vocab` is the
final norm, `lm_head` and the scale, so a logit lens on the last block equals `logits` exactly.
A direct logit attribution through `lm_head` alone is off by that constant factor.

```python
with model.trace(prompt):
    resid = model.layers[-1].layer_output.save()
    raw = model.lm_head.output.save()
    logits = model.logits.save()

assert torch.equal(logits, raw * model.config.logit_scale)
assert torch.equal(model.project_on_vocab(resid), logits)
```

## The embeddings are not scaled

`embed_tokens` is a plain lookup over 256,000 tokens, so `token_embeddings` equals
`layers[0].input`. Because the weight is shared with `lm_head`, an edit to `embed_tokens.weight`
also edits the unembedding.
"""
