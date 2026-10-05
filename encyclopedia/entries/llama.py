"""Llama: the block the standard names are taken from."""

MODEL_TYPE = "llama"
TITLE = "Llama"
SUBTITLE = (
    "The reference block: one RMSNorm before each sublayer and none after, so what reaches the "
    "residual stream is the attention's and the MLP's own output, with rotary positions applied to "
    "queries and keys inside the attention."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "meta-llama/Llama-3.1-8B"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-LlamaForCausalLM"
CHECKPOINTS = [
    "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-70b-hf",
    "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-70B",
    "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B", "meta-llama/Llama-3.1-405B",
    "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
]

VLLM = True
QUIRKS: list[str] = []

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
            "detail": "{num_heads} query, {num_kv_heads} key/value heads",
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
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "A plain lookup: no scale and no position embedding is added, so token_embeddings equals "
             "layers[0].input. Positions enter as a rotation of queries and keys inside each attention.",
    "norm": "RMSNorm whose gain is norm.weight as stored.",
    "head": "Untied on 8B, 70B and Llama-2-7B; tied to embed_tokens on Llama-3.2-1B and 3B (tie_word_embeddings). "
            "Nothing follows it: logits equals lm_head.output, and project_on_vocab is lm_head(norm(hidden)).",
}

NOTES = """
## The block, in order

```
h   = x + self_attn(input_layernorm(x))
out = h + mlp(post_attention_layernorm(h))
```

Two RMSNorms per block, each before its sublayer, none after. `post_attention_layernorm` is
named for where it sits on the stream, after the attention's add: it is the MLP's input norm.
Some families with a norm after each sublayer give the name to a different module (on Gemma-2
it is the norm after the attention), so check the block's forward before porting a hook by name.

## The contributions are the modules' outputs

`attention_output` is `self_attn.output[0]` and `mlp_output` is `mlp.output`: the same tensors,
added to the stream with nothing in between. The identity holds exactly, in float32 on the
pinned checkpoint and in bfloat16 on Llama-3.2-1B:

```python
with model.trace(prompt):
    x = model.layers[5].input.save()
    attn = model.layers[5].self_attn.attention_output.save()
    mlp = model.layers[5].mlp.mlp_output.save()
    out = model.layers[5].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)
```

So scaling a module's output scales its contribution: `model.layers[i].mlp.output[:] *= 0.5`
and `model.layers[i].mlp.mlp_output[:] *= 0.5` give the same logits. The inputs are the norms'
outputs: `self_attn.input` is `input_layernorm.output`, `mlp.input` is
`post_attention_layernorm.output`. The stream between the two adds has no standard value; it is
`model.layers[i].post_attention_layernorm.input`, which equals
`model.layers[i].input + model.layers[i].self_attn.attention_output`.

## The default load runs the same attention as eager

A default load runs `sdpa`, and the six attention interior values report unavailable;
`attn_implementation="eager"` serves them. Llama has no score softcap, window or sink, so the two
paths compute the same function: on Llama-3.2-1B in float32 their logits differ by at most
`2e-5`, and in bfloat16 by up to `0.25` with the same top token at every position of a test prompt. The query
scale is `head_dim ** -0.5` (`128 ** -0.5` on 8B, `64 ** -0.5` on 3.2-1B).

## Grouped-query attention on the 3.x sizes

8B and 3.2-1B have 32 query heads over 8 key/value heads, 3.2-3B 24 over 8, 70B 64 over 8.
`attention_keys` and `attention_values` are read before `repeat_kv`, `[batch, 8, seq, head_dim]`,
and query head `h` reads key/value head `h // 4` on 8B: an edit to key/value head `j` reaches
query heads `4j` to `4j + 3`, a contiguous block. Llama-2-7B, Code Llama 7B and the pinned
checkpoint have one key/value head per query head, so the same code edits one head there.

```python
attn = model.layers[5].self_attn
groups = model.num_heads // model.num_kv_heads   # 4 on 8B and 3.2-1B
with model.trace(prompt):
    attn.attention_values[:, 1] = 0             # key/value head 1
    heads = attn.attention_head_outputs.save()  # only query heads 4 to 7 change
```

## Queries and keys are read after the rotary embedding

`attention_queries` and `attention_keys` are rotated by position; the projections before it are
the outputs of `q_proj` and `k_proj`. The two agree at position 0, where the rotation is the
identity, and nowhere else. A key or query moved to another position keeps the rotation of the
position it was read at. The cosines and sines are computed once per forward by the native
`model.model.rotary_emb` and passed to every block; Llama-3.1, 3.2 and 3.3 use `rope_type`
`llama3` (`rope_theta` 500000, rescaled frequencies), Llama-2 the plain rotary with
`rope_theta` 10000. To read both sides in one trace, take the projection through the attention's
`source`: on the first trace of a freshly loaded model, reading `q_proj.output` before an
interior value raises `OutOfOrderError`.

```python
attn = model.layers[5].self_attn
with model.trace(prompt):
    q_raw = attn.source.self_q_proj_0.output.save()   # before the rotary
    q = attn.attention_queries.save()                 # after it
```

## Position 0 is a sink with a very large norm

On Llama-3.2-1B the residual stream at position 0 has a norm near 420 after blocks 1 and 5,
against a mean of 2.4 and 4.5 at the other positions, and heads put on average 0.67 to 0.87 of their attention
on it (blocks 0, 1, 5, 8 and 15, on a test prompt). The Llama-3 tokenizer prepends its beginning-of-text token
(id `128000`); without it the first real token takes the role (norm 865 after block 5). A mean over
positions that includes position 0 is dominated by it: slice `[:, 1:]` before averaging
activations for steering vectors, mean ablation or probes.

```python
with model.trace(prompt):
    resid = model.layers[5].layer_output.save()

resid[0].float().norm(dim=-1)        # 422 at position 0, 4 to 5 elsewhere on 3.2-1B
```

## Target tokens differ by tokenizer

On Llama-3, `"Paris"` (`60704`) and `" Paris"` (`12366`) are two whole-word tokens, and
`get_first_tokens(["Paris"], model)` returns both. Llama-2's SentencePiece tokenizer prefixes
the word marker itself: `"Paris"` and `" Paris"` both encode to `▁Paris` (`3681`), and
`get_first_tokens(["Paris"], model)` returns `[2177, 3681]`, where `2177` is the mid-word
fragment `Par`. Decode the target ids on the checkpoint you run.

## The readout and the embeddings

`model.logits` is `model.lm_head.output`: no cap and no scale, so `project_on_vocab` is
`lm_head(norm(hidden))` and a logit lens at the last block equals `logits`. The final norm's
gain is `model.norm.weight` as stored. Llama-3.2-1B and 3B tie `lm_head` to `embed_tokens`
(one parameter), so an edit to `embed_tokens.weight` is an edit to the unembedding; 8B, 70B and
Llama-2-7B do not.

## What this family module covers

The module serves every checkpoint whose config says `model_type` `llama`: besides Meta's
Llama-2, Llama-3, 3.1, 3.2 and 3.3, that includes Code Llama, TinyLlama,
DeepSeek-R1-Distill-Llama-8B, SmolLM2 and Helium-1-2B. Their settings differ: SmolLM2-135M
ties its embeddings, has 9 query heads over 3 key/value heads, and its tokenizer adds no
beginning-of-sequence token; Code Llama 7B has `rope_theta` 1000000. Llama 4 is the
`llama4_text` family.

## Sparse autoencoders and transcoders

- **Llama Scope** (`OpenMOSS-Team/Llama-Scope`), on Llama-3.1-8B, every block. The `R` SAEs read
  the stream after the block, `model.layers[i].layer_output`; `A` reads
  `model.layers[i].self_attn.attention_output`; `M` reads `model.layers[i].mlp.mlp_output`. The
  `TC` transcoders read the normed stream, `model.layers[i].mlp.input`, and predict `mlp_output`.
  Inputs are scaled to a norm of `sqrt(hidden_size)` before encoding.
- **EleutherAI** `EleutherAI/sae-llama-3-8b-32x`, on Meta-Llama-3-8B: residual-stream SAEs whose
  hook points `layers.i` are the block outputs, `model.layers[i].layer_output`, and `embed_tokens`
  is `model.token_embeddings`.
- **circuit-tracer** transcoders on Llama-3.2-1B, per-layer (`mntss/transcoder-Llama-3.2-1B`)
  and cross-layer (`mntss/clt-llama-3.2-1b-524k`). They read `hook_resid_mid`, the stream before
  the MLP's norm, `model.layers[i].post_attention_layernorm.input`, and write `hook_mlp_out`,
  `model.layers[i].mlp.mlp_output`:

```python
with model.trace(prompt):
    mid = model.layers[i].post_attention_layernorm.input.save()   # hook_resid_mid
    out = model.layers[i].mlp.mlp_output.save()                    # hook_mlp_out
```
"""
