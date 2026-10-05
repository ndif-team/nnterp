"""Granite: Llama's block with four config scalars around it; the first of the Granite lineage."""

MODEL_TYPE = "granite"
TITLE = "Granite"
SUBTITLE = (
    "Llama's block with four config scalars around it: the embeddings are multiplied before block 0, "
    "each sublayer's output is multiplied by residual_multiplier before it is added, the attention "
    "scales its scores by attention_multiplier, and the logits are the head's output divided by logits_scaling."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "ibm-granite/granite-3.0-2b-base"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-GraniteForCausalLM"
CHECKPOINTS = [
    "ibm-granite/granite-3.0-2b-base", "ibm-granite/granite-3.0-2b-instruct",
    "ibm-granite/granite-3.0-8b-base", "ibm-granite/granite-3.0-8b-instruct",
    "ibm-granite/granite-3.1-2b-base", "ibm-granite/granite-3.1-2b-instruct",
    "ibm-granite/granite-3.1-8b-base", "ibm-granite/granite-3.1-8b-instruct",
    "ibm-granite/granite-3.2-2b-instruct", "ibm-granite/granite-3.2-8b-instruct",
    "ibm-granite/granite-3.3-2b-base", "ibm-granite/granite-3.3-2b-instruct",
    "ibm-granite/granite-3.3-8b-base", "ibm-granite/granite-3.3-8b-instruct",
    "ibm-granite/granite-4.1-3b-base", "ibm-granite/granite-4.1-3b",
    "ibm-granite/granite-4.1-8b-base", "ibm-granite/granite-4.1-8b",
    "ibm-granite/granite-4.1-30b-base", "ibm-granite/granite-4.1-30b",
    "ibm-granite/granite-4.2-3b", "ibm-granite/granite-4.2-8b", "ibm-granite/granite-4.2-30b",
]

#: `granite` hashes to 156, beside Gemma's 145 and 157; the Granite lineage sits at 205.
PALETTE = {"hue": 205}
VLLM = False
QUIRKS = ["scaled-residual-adds", "embedding-multiplier", "scaled-logits"]

#: What the visualization draws: the sublayers in forward order, each with its norms,
#: the standard value that is its contribution, and the values read inside it.
#: ``detail`` is formatted with the sizes and the config keys.
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
            "detail": "{num_heads}/{num_kv_heads} heads, scale {attention_multiplier}",
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
                     "residual_multiplier (0.22 on 2B), so the plain sum is exact, in float32 and bfloat16 alike.",
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "token_embeddings is the lookup itself; the model multiplies it by embedding_multiplier "
             "(12.0 on 3.x and 4.1) before block 0, so layers[0].input is token_embeddings · 12.",
    "layers": "Each block adds its attention's and its MLP's output times residual_multiplier (0.22 on 2B).",
    "head": "lm_head shares its weight with embed_tokens (tie_word_embeddings) on Granite 3.x and 4.1; "
            "Granite 4.2 unties it.",
    "logits": "logits = lm_head.output / logits_scaling (8.0 on 2B). project_on_vocab divides too.",
}

NOTES = """
## The block, in order

```
x      = embed_tokens(ids) * embedding_multiplier
h      = x + self_attn(input_layernorm(x)) * residual_multiplier
out    = h + mlp(post_attention_layernorm(h)) * residual_multiplier
logits = lm_head(norm(out)) / logits_scaling
```

The modules and their names are Llama's, `post_attention_layernorm` included (the MLP's input
norm); the four scalars sit between them and are read from the config, so each one is a plain
number, the same on every block. On the released checkpoints:

- 3.x 2B: `embedding_multiplier` 12, `residual_multiplier` 0.22, `attention_multiplier` 1/64, `logits_scaling` 8.
- 3.x 8B and 4.1-8B: 12, 0.22, 1/128, 16.
- 4.1-3B: 12, 0.22, 1/64, 10.
- 4.1-30B: 12, 0.175, 1/128, 16.
- 4.2 (3B, 8B, 30B): 1, 1, `1 / head_dim`, 1, with `lm_head` untied. On 4.2 the contributions equal the
  modules' outputs and `logits` equals `lm_head.output`.

## The contributions are scaled copies of the modules' outputs

`attention_output` is `self_attn.output[0] * residual_multiplier` and `mlp_output` is
`mlp.output * residual_multiplier`: the tensors the block adds, computed from the modules' outputs,
which are not themselves on the stream. With them the plain identity holds bit for bit on 2B, in
float32 and bfloat16:

```python
with model.trace(prompt):
    x = model.layers[1].input.save()
    attn_raw = model.layers[1].self_attn.output[0].save()
    attn = model.layers[1].self_attn.attention_output.save()
    mlp = model.layers[1].mlp.mlp_output.save()
    out = model.layers[1].layer_output.save()

m = model.config.residual_multiplier                  # 0.22
torch.testing.assert_close(attn, attn_raw * m)
torch.testing.assert_close(x + attn + mlp, out)
```

Direct logit attribution and any decomposition of the stream into sublayer terms use
`attention_output` and `mlp_output`; `self_attn.output[0]` and `mlp.output` are 1/0.22 ≈ 4.5 times
larger than what reaches the stream. A vector added to `mlp_output` reaches the stream as it is; the
same vector added to `mlp.output` reaches it times 0.22. A read changes nothing: the forward stays
bit-identical.

## A write to a contribution rounds every position

A write to `attention_output` or `mlp_output` reaches the model by dividing the whole edited copy
by `residual_multiplier` and handing it back as the module's output, and the block multiplies it
again. Positions you did not edit go through that round trip too. On 2B in bfloat16, adding a
unit-norm vector to `mlp_output` at the last position of block 1 moved the logits at every earlier
position, which the edit cannot reach causally, by up to 0.19 (a KL of up to 0.002), while its own
effect at the edited position was a KL of 8e-5. In float32 the stray change is 2e-5 in the logits.
Editing the module's output by the vector over the multiplier gives the same effect at the edited
position, up to rounding, and leaves every other position bit-identical in either dtype:

```python
m = model.config.residual_multiplier
with model.trace(prompt):
    model.layers[1].mlp.mlp_output[:, -1] += v    # every position rounds

with model.trace(prompt):
    model.layers[1].mlp.output[:, -1] += v / m    # position -1 only
```

For fine-grained edits (one position, one head, a small steering vector), edit the module's output
this way or load in float32. The 3.0 base checkpoints (2B, 8B) are stored in float32, the others
in bfloat16.

## The query scale is attention_multiplier, 1/head_dim

The attention multiplies `q · kᵀ` by `config.attention_multiplier` in place of `head_dim ** -0.5`.
On every released checkpoint it is `1 / head_dim` (1/64 on 2B), an eighth of Llama's scale at
`head_dim` 64. `attention_scores` are read after the scale and the causal mask, and
`softmax(attention_scores)` reproduces `attention_probabilities` exactly in float32. Code that
recomputes scores from queries and keys takes the multiplier from the config:

```python
attn = model.layers[1].self_attn
with model.trace(prompt):
    q = attn.attention_queries.save()
    k = attn.attention_keys.save()
    scores = attn.attention_scores.save()

k = k.repeat_interleave(model.num_heads // model.num_kv_heads, dim=1)
scale = model.config.attention_multiplier    # not head_dim ** -0.5
qk = q @ k.transpose(-1, -2) * scale
causal = torch.ones_like(qk[0, 0], dtype=torch.bool).tril()
torch.testing.assert_close(qk[..., causal], scores[..., causal])
```

The 2B has 32 query heads over 8 key/value heads: `attention_keys` and `attention_values` are
`[batch, 8, seq, 64]`, read before `repeat_kv`, and an edit to key/value head `j` reaches query heads
`4j` to `4j + 3`. A default load runs `sdpa`, which computes the same function: on 2B its logits
equal eager's bit for bit in float32, and differ by up to 0.23 in bfloat16 with the same top token.
The six interior values need `attn_implementation="eager"`.

## Logits are the head's output divided by logits_scaling

`model.logits` is `model.lm_head.output / logits_scaling`, exactly, and `project_on_vocab` divides
too, so a logit lens at the last block equals `logits`. A softmax over `lm_head.output`, or over
`lm_head(norm(hidden))` written by hand, is at temperature 1/8 on 2B: on a test prompt the top
token's probability is 0.94 from `logits` and 1.00 from `lm_head.output`. Take probabilities and KLs
from `logits` or `project_on_vocab`.

```python
with model.trace(prompt):
    resid = model.layers[-1].layer_output.save()
    raw = model.lm_head.output.save()
    logits = model.logits.save()

torch.testing.assert_close(logits, raw / model.config.logits_scaling)
torch.testing.assert_close(model.project_on_vocab(resid), logits)
```

`lm_head` and `embed_tokens` are one parameter on 3.x and 4.1, so an edit to
`embed_tokens.weight` is an edit to the unembedding.

## token_embeddings is the unscaled lookup

The embedding module returns the plain lookup, and the model multiplies it by
`embedding_multiplier` afterwards, so `token_embeddings` is twelve times smaller than
`layers[0].input` (row norms about 0.9 against 10.7 on 2B). Embeddings passed as `inputs_embeds`
are multiplied too. An edit to `token_embeddings` reaches block 0 times 12; to set what block 0
reads, edit `layers[0].input`.

```python
with model.trace(prompt):
    emb = model.token_embeddings.save()
    first = model.layers[0].input.save()

torch.testing.assert_close(first, emb * model.config.embedding_multiplier)
```

## Position 0 is a sink, and no BOS token is added

Neither tokenizer prepends a beginning-of-sequence token, so the first real token takes the
role. On 2B its residual norm is 293 after block 2, against 8 to 12 at the other positions of a
test prompt, and about 4000 after block 20, against a mean of 16; in blocks 5, 10 and 20 heads put
0.71 to 0.83 of their attention on it (two test prompts). 4.1-3B shows the same at block 5 (norm
107, 0.67 of the attention). A later token can grow too: ` the` reached 243 after block 5 in one
prompt. Slice `[:, 1:]` before averaging activations for steering vectors, mean ablation or probes.

```python
with model.trace(prompt):
    resid = model.layers[2].layer_output.save()

resid[0].norm(dim=-1)        # 293 at position 0, 8 to 12 elsewhere on 2B
```

## Target tokens differ by tokenizer

The 3.x tokenizer (49152 tokens) splits `" Paris"` into `ĠPar` and `is`; the 4.1 tokenizer
(100352 tokens) has `" Paris"` (`12366`) and `"Paris"` (`60704`) as whole tokens. Decode the
target ids on the checkpoint you run.

## What this family module covers

The module serves every checkpoint whose config says `model_type` `granite`: the dense Granite 3.0,
3.1, 3.2 and 3.3 at 2B and 8B, and Granite 4.1 and 4.2 at 3B, 8B and 30B. Granite-7B and the
Granite Code 3B and 8B are `llama`, Granite Code 20B and 34B `gpt_bigcode`. The mixtures of
experts (3.x 1B-A400M, 3B-A800M) are `granitemoe`, all of Granite 4.0 including the dense micro,
1B and 350M is `granitemoehybrid`, and the Swash models are `granite_swa` and `granitemoe_swa`.
"""
