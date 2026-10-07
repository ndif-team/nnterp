"""Granite SWA: Granite's scaled block with sliding-window attention and a per-head sink that scales the head outputs."""

MODEL_TYPE = "granite_swa"
TITLE = "Granite Swash"
SUBTITLE = (
    "Granite's block and multipliers with sliding-window attention on most blocks and a learned sink per head "
    "that scales each head's output after the softmax, so the pattern's rows sum to one and the head outputs do not "
    "carry all of it."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "ibm-granite/granite-swash-2b"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-GraniteSWAForCausalLM"
CHECKPOINTS = ["ibm-granite/granite-swash-2b"]

#: Granite lineage (205, set by granite); Granite SWA sits beside it.
PALETTE = {"hue": 199}
VLLM = False
QUIRKS = ["scaled-residual-adds", "sliding-window", "embedding-multiplier", "scaled-logits"]

#: Numbers in the notes come from granite-swash-2b's weights in bfloat16 on an RTX A6000; the snippets also ran
#: on the pinned tiny checkpoint, and on a copy of it with Granite's multipliers away from 1.0.
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
            "detail": "{num_heads}/{num_kv_heads} heads, scale {attention_multiplier}, sink",
            "variants": {
                "sliding_attention": "window of {sliding_window} tokens, sink",
                "full_attention": "full causal attention, sink",
            },
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "Named for its place on the stream, after the attention's add: it is the MLP's input norm, "
                             "and its input is the stream between the two adds.",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
    ],
    "identity_note": "The terms are already scaled: attention_output and mlp_output are the modules' outputs times "
                     "residual_multiplier (0.28 on 2B), so the plain sum is exact, in float32 and bfloat16 alike.",
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "token_embeddings is the lookup itself; the model multiplies it by embedding_multiplier (12) before block 0, "
             "so layers[0].input is token_embeddings · 12.",
    "layers": "Each block adds its attention's and its MLP's output times residual_multiplier (0.28 on 2B).",
    "head": "lm_head shares its weight with embed_tokens (tie_word_embeddings).",
    "logits": "logits = lm_head.output / logits_scaling (10 on 2B). project_on_vocab divides too.",
}

NOTES = """
## The block, in order

```
x      = embed_tokens(ids) * embedding_multiplier
h      = x + self_attn(input_layernorm(x)) * residual_multiplier
out    = h + mlp(post_attention_layernorm(h)) * residual_multiplier
logits = lm_head(norm(out)) / logits_scaling
```

Granite's block, names and multipliers, with its own attention. On granite-swash-2b
`embedding_multiplier` is 12, `residual_multiplier` 0.28, `attention_multiplier` 1/128 and
`logits_scaling` 10. Of its 24 blocks, 7 attend over the whole prefix (0, 3, 7, 11, 15, 19 and
23) and 17 over the last 128 tokens; `config.layer_types` says which, and
`layers[i].self_attn._module.sliding_window` is 128 on a sliding block and `None` on a full one.
Every block applies rotary embeddings with base 10000.

## The sink scales the head outputs, not the pattern

Each head has a learned sink, `self_attn._module.sinks`. The attention takes the softmax over the
real keys, mixes the values with it, and then multiplies each query's head output by
`sigmoid(logsumexp(scores) - sink)`: the share of the mass a softmax with the sink as one more key
would leave on the real keys. So `attention_probabilities` rows sum to one,
`softmax(attention_scores)` reproduces them, and `attention_head_outputs` are read after the scale.

```python
attn = model.layers[1].self_attn
with model.trace(prompt):
    v = attn.attention_values.save()
    scores = attn.attention_scores.save()
    probs = attn.attention_probabilities.save()
    heads = attn.attention_head_outputs.save()     # [batch, seq, heads, head_dim]

share = torch.sigmoid(scores.logsumexp(-1) - attn._module.sinks.view(1, -1, 1))
v = v.repeat_interleave(model.num_heads // model.num_kv_heads, dim=1)
torch.testing.assert_close(scores.softmax(-1), probs)
torch.testing.assert_close(((probs @ v) * share[..., None]).transpose(1, 2), heads)
```

The share is small on most blocks. On a code prompt its median over heads and queries is 0.68 in
block 0 and between 0.05 and 0.37 in every other block, and its largest value in a block is 0.67
to 1.00: most queries of most heads write a fraction of what the pattern alone would give. A pattern read as "how much
this head moves from token j" is `probs * share[..., None]`. An edit to the pattern does not change
the share, which is computed from the scores; to set what a head writes, edit
`attention_head_outputs`.

## Eager is the default

transformers' Granite SWA has no `sdpa` path, so a default load runs the eager attention and the
six interior values are available without `attn_implementation="eager"`.

The query scale is `config.attention_multiplier`, 1/128, which is `1 / head_dim` (Llama's would be
`128 ** -0.5`). The 2B has 20 query heads over 4 key/value heads of 128: an edit to key/value head
`j` reaches query heads `5j` to `5j + 4`.

## The contributions are scaled copies of the modules' outputs

`attention_output` is `self_attn.output[0] * residual_multiplier` and `mlp_output` is
`mlp.output * residual_multiplier`, so the plain identity holds, bit for bit in bfloat16 on 2B:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn_raw = model.layers[1].self_attn.output[0].save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

m = model.config.residual_multiplier                  # 0.28
torch.testing.assert_close(attn, attn_raw * m)
torch.testing.assert_close(x + attn + mlp, out)
```

A write to a contribution reaches the model as the whole edited copy divided by
`residual_multiplier`, which the block multiplies again. With 0.28 that round trip is exact in
bfloat16: every product `x * 0.28` of a bfloat16 `x` comes back from `/ 0.28 * 0.28` unchanged, so
the positions you did not edit come back bit-identical. Adding a
unit vector to `mlp_output` or `attention_output` at the last position of block 1, 12 or 23 left
every earlier position's logits unchanged. Editing `mlp.output[:, -1] += v / m` has the same effect.

## Logits are divided by 10, and the embeddings multiplied by 12

`model.logits` is `model.lm_head.output / logits_scaling` exactly, and `project_on_vocab` divides
too, so a logit lens at the last block equals `logits`. A softmax over `lm_head.output` is at
temperature 1/10: after "The capital of the United Kingdom is", ` London` has probability 0.79
from `logits` and 1.00 from `lm_head.output`. `token_embeddings` is the plain lookup, and
`layers[0].input` is it times 12; to set what block 0 reads, edit `layers[0].input`.

```python
with model.trace(prompt):
    emb = model.token_embeddings.save()
    first = model.layers[0].input.save()
    resid = model.layers[-1].layer_output.save()
    raw = model.lm_head.output.save()
    logits = model.logits.save()

torch.testing.assert_close(first, emb * model.config.embedding_multiplier)
torch.testing.assert_close(logits, raw / model.config.logits_scaling)
torch.testing.assert_close(model.project_on_vocab(resid), logits)
```

`lm_head` and `embed_tokens` are one parameter, so an edit to `embed_tokens.weight` is an edit to
the unembedding.

## The tokenizer

The vocabulary is Granite 4's, 100352 tokens. No beginning-of-sequence token is prepended.
`" Paris"` (12366), `"Paris"` (60704) and `" London"` (7295) are single tokens.

## What this family module covers

Every checkpoint whose config says `granite_swa`: granite-swash-2b. The module also serves a
block without rotary (`layer_rope_theta` 0 on that block), which no released checkpoint uses.
`granitemoe_swa` is the Swash mixture, the same attention in a block whose MLP is a mixture of
experts.
"""
