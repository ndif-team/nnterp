"""HyperCLOVA X: a sandwich block (post_norm1, post_norm2) with Granite's four config multipliers around it."""

MODEL_TYPE = "hyperclovax"
TITLE = "HyperCLOVA X"
SUBTITLE = (
    "A sandwich block whose post-norms keep their native names, post_norm1 and post_norm2, with four config "
    "scalars around it: the embeddings and each post-norm's output are multiplied before they reach the "
    "stream, the scores are scaled by attention_multiplier, and the logits are multiplied by logits_scaling."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "naver-hyperclovax/HyperCLOVAX-SEED-Think-14B"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-HyperCLOVAXForCausalLM"
CHECKPOINTS = ["naver-hyperclovax/HyperCLOVAX-SEED-Think-14B"]

#: Set by hues.py (no kin, in a gap between lineages).
PALETTE = {"hue": 273}
VLLM = False
QUIRKS = ["sandwich-norms", "scaled-residual-adds", "embedding-multiplier", "scaled-logits"]

#: No checkpoint of this family is small enough to run here: the identities, read orders and the write
#: round trip were checked on the pinned tiny checkpoint and on the suite's copy of it with
#: residual_multiplier 0.22, logits_scaling 4 and embedding_multiplier 3, in float32 and bfloat16; the
#: numbers for Think-14B come from its config.

BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "post_norm": "post_norm1",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads}/{num_kv_heads} heads, scale {attention_multiplier}",
            "post_norm_note": "The norm after the attention. attention_output is its output times residual_multiplier.",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "post_norm": "post_norm2",
            "contribution": "mlp_output",
            "detail": "SwiGLU: {hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
            "pre_norm_note": "Named for its place on the stream, after the attention's add: it is the MLP's input norm. "
                             "The norm after the attention is post_norm1.",
            "post_norm_note": "The norm after the MLP. mlp_output is its output times residual_multiplier.",
        },
    ],
    "identity_note": "The terms are already scaled: attention_output and mlp_output are the post-norms' outputs "
                     "times residual_multiplier (1.0 on Think-14B), so the plain sum is exact, in float32 and bfloat16 alike.",
}

STRIP = {
    "embed": "token_embeddings is the lookup itself; the model multiplies it by embedding_multiplier "
             "(10.0 on Think-14B) before block 0, so layers[0].input is token_embeddings · 10.",
    "layers": "Each block adds its post-norms' outputs times residual_multiplier (1.0 on Think-14B).",
    "head": "lm_head has its own weight: tie_word_embeddings is false.",
    "logits": "logits = lm_head.output · logits_scaling (0.125 on Think-14B). project_on_vocab multiplies too.",
}

NOTES = """
## The block, in order

```
x      = embed_tokens(ids) * embedding_multiplier
h      = x + post_norm1(self_attn(input_layernorm(x))) * residual_multiplier
out    = h + post_norm2(mlp(post_attention_layernorm(h))) * residual_multiplier
logits = lm_head(norm(out)) * logits_scaling
```

Four RMSNorms per block, two per sublayer, and the names are the trap: `post_attention_layernorm`
is the MLP's *input* norm, as on Llama, and the norms after the sublayers are `post_norm1` (after
the attention) and `post_norm2` (after the MLP). nnterp keeps the native names. On Think-14B the
four scalars are `embedding_multiplier` 10, `residual_multiplier` 1, `attention_multiplier` 1/128
and `logits_scaling` 0.125, each one number for the whole model.

## The contributions are the post-norms' outputs, times residual_multiplier

`attention_output` is `post_norm1.output * residual_multiplier` and `mlp_output` is
`post_norm2.output * residual_multiplier`: the tensors the block adds. On Think-14B the multiplier is
1.0, so they equal the post-norms' outputs bit for bit. `self_attn.output[0]` and `mlp.output` are
the tensors *entering* the post-norms. The identity is the plain sum, exact in float32 and bfloat16:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    post = model.layers[1].post_norm1.output.save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

torch.testing.assert_close(attn, post * model.config.residual_multiplier)
torch.testing.assert_close(x + attn + mlp, out)
```

**RMSNorm is scale-invariant, so scale the contribution, not the module.** `post_norm2(0.5 * y)`
equals `post_norm2(y)` up to `eps`: halving `mlp.output` leaves the stream as it was, and zeroing
it works only because the norm of zero is zero. Partial ablations and steering go through
`mlp_output` / `attention_output`, which the block adds as they are:

```python
with model.trace(prompt):
    model.layers[1].mlp.mlp_output[:] *= 0.5        # halves what the block adds

with model.trace(prompt):
    model.layers[1].mlp.output[:] *= 0.5            # a no-op: post_norm2 rescales it back
```

## A write to a contribution goes through the multiplier

The contributions are computed copies: a write to `attention_output` or `mlp_output` reaches the
model by dividing the whole edited copy by `residual_multiplier` and handing it back as the
post-norm's output, and the block multiplies it again. With the multiplier at 1.0, as on Think-14B,
both steps are exact: an edit at one position leaves every other position bit-identical, in
bfloat16 too. With another multiplier the unedited positions go through the round trip as well and
move by rounding in bfloat16. Editing the
post-norm's output by the vector over the multiplier reaches the same place and touches nothing
else, for any multiplier:

```python
m = model.config.residual_multiplier
with model.trace(prompt):
    model.layers[1].mlp.mlp_output[:, -1] += v            # exact when m is 1.0

with model.trace(prompt):
    model.layers[1].post_norm2.output[:, -1] += v / m     # position -1 only, for any m
```

A read changes nothing: the forward stays bit-identical.

## The query scale is attention_multiplier, 1/head_dim

The attention multiplies `q · kᵀ` by `config.attention_multiplier` in place of `head_dim ** -0.5`.
On Think-14B it is 1/128, `1 / head_dim`, a factor of 11.3 below `128 ** -0.5`. `attention_scores`
are read after the scale and the causal mask, and `softmax(attention_scores)` reproduces
`attention_probabilities`. Code that recomputes scores takes the multiplier from the config:

```python
attn = model.layers[1].self_attn
with model.trace(prompt):
    q = attn.attention_queries.save()
    k = attn.attention_keys.save()
    scores = attn.attention_scores.save()

k = k.repeat_interleave(model.num_heads // model.num_kv_heads, dim=1)
qk = q @ k.transpose(-1, -2) * model.config.attention_multiplier    # not head_dim ** -0.5
causal = torch.ones_like(qk[0, 0], dtype=torch.bool).tril()
torch.testing.assert_close(qk[..., causal], scores[..., causal])
```

Think-14B has 48 query heads over 8 key/value heads: `attention_keys` and `attention_values` are
`[batch, 8, seq, 128]`, read before `repeat_kv`, and an edit to key/value head `j` reaches query
heads `6j` to `6j + 5`. The six interior values need `attn_implementation="eager"`.

## Logits are the head's output times logits_scaling

`model.logits` is `model.lm_head.output * logits_scaling`, exactly, 0.125 on Think-14B, and
`project_on_vocab` multiplies too, so a logit lens at the last block equals `logits`. A softmax over
`lm_head.output`, or over `lm_head(norm(hidden))` written by hand, is at temperature 1/8: sharper
than the model's own distribution. Take probabilities and KLs from `logits` or
`project_on_vocab`.

```python
with model.trace(prompt):
    resid = model.layers[-1].layer_output.save()
    raw = model.lm_head.output.save()
    logits = model.logits.save()

torch.testing.assert_close(logits, raw * model.config.logits_scaling)
torch.testing.assert_close(model.project_on_vocab(resid), logits)
```

`lm_head` and `embed_tokens` are separate weights.

## token_embeddings is the unscaled lookup

The embedding module returns the plain lookup, and the model multiplies it by
`embedding_multiplier` afterwards: `layers[0].input` is `token_embeddings * 10` on Think-14B.
Embeddings passed as `inputs_embeds` are multiplied too. An edit to `token_embeddings` reaches
block 0 times 10; to set what block 0 reads, edit `layers[0].input`.

```python
with model.trace(prompt):
    emb = model.token_embeddings.save()
    first = model.layers[0].input.save()

torch.testing.assert_close(first, emb * model.config.embedding_multiplier)
```

## Loading Think-14B

The checkpoint is stored in float32 (`torch_dtype` in its config), and a default load keeps it, about
59 GB for its 14.7B parameters; pass `dtype=torch.bfloat16` to halve it. The tokenizer prepends no
BOS token to a plain prompt; the chat template starts with `<|im_start|>`.

## The 32B vision-language checkpoint

`HyperCLOVAX-SEED-Think-32B` is a vision-language wrapper (`hyperclovax_vision_v2`, remote code) whose
text model is this family. Its text config turns the post-norms off (`use_post_norm` false), sets
`embedding_multiplier`, `residual_multiplier` and `logits_scaling` to 1.0 and `attention_multiplier`
to `128 ** -0.5`: under these names it computes Llama's block. With `use_post_norm` false, `post_norm1`
and `post_norm2` are identities, and `attention_output` and `mlp_output` are the modules' outputs
times `residual_multiplier`.
"""
