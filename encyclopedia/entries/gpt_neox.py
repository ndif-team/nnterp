"""GPT-NeoX: Pythia, GPT-NeoX-20B, and the other checkpoints that load as GPTNeoXForCausalLM."""

MODEL_TYPE = "gpt_neox"
TITLE = "GPT-NeoX"
SUBTITLE = (
    "Pythia's block is parallel: attention and MLP each read their own LayerNorm of the block input and "
    "the block adds both at once, so a block's MLP never sees that block's attention; queries, keys and "
    "values come from one fused projection, and rotary turns a quarter of each head."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "EleutherAI/pythia-70m-deduped"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"
CHECKPOINTS = [
    "EleutherAI/pythia-14m-deduped", "EleutherAI/pythia-14m",
    "EleutherAI/pythia-31m-deduped", "EleutherAI/pythia-31m",
    "EleutherAI/pythia-70m-deduped", "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m-deduped", "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m-deduped", "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b-deduped", "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b-deduped", "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b-deduped", "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b-deduped", "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b-deduped", "EleutherAI/pythia-12b",
    "EleutherAI/gpt-neox-20b",
]

VLLM = True
QUIRKS = ["parallel-blocks", "fused-qkv", "partial-rotary"]

#: What the visualization draws: Pythia's parallel block, two pre-norms of the block input,
#: no post-norms; the two contributions join the stream in one add.
BLOCK = {
    "topology": "parallel",
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
            "detail": "{num_heads} heads × {head_dim}, fused qkv",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "Despite its name this norm does not follow the attention. Under use_parallel_residual "
                             "(every Pythia checkpoint) it normalizes the block input, the same tensor input_layernorm "
                             "reads, with its own weight and bias.",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {hidden_act}",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "embed_in is a plain lookup: no scale and no position embedding (position enters through rotary in "
             "each attention), so token_embeddings equals layers[0].input.",
    "norm": "final_layer_norm is a LayerNorm with a bias; project_on_vocab applies it, bias included.",
    "head": "lm_head has its own weight, not tied to embed_in (tie_word_embeddings is false), and no bias. "
            "Nothing follows it: logits equals lm_head.output.",
}

NOTES = """
## The block is parallel on Pythia

```
out = x + self_attn(input_layernorm(x)) + mlp(post_attention_layernorm(x))
```

Both LayerNorms read the block input `x`, each with its own weight and bias, so
`self_attn.input` and `mlp.input` are two different normalizations of the same tensor. The name
`post_attention_layernorm` is the trap: on this block it does not follow the attention. Every
Pythia config sets `use_parallel_residual: true`, and `EleutherAI/gpt-neox-20b` leaves it out and
takes the config default, also `true`.

With `use_parallel_residual: false` the same modules run in sequence, as on Llama:

```
h   = x + self_attn(input_layernorm(x))
out = h + mlp(post_attention_layernorm(h))
```

`togethercomputer/RedPajama-INCITE-Base-3B-v1` is one such checkpoint. The diagram draws
Pythia's setting; read `model.config.use_parallel_residual` before reusing a recipe on another
`GPTNeoXForCausalLM` checkpoint.

## A block's MLP never sees its own attention

The contribution identity holds in both settings: `attention_output` is the attention module's
output (`dense`, bias included) and `mlp_output` the MLP's. On the parallel block nothing
connects the two sublayers of one block: an edit to `attention_output` leaves that block's
`mlp.input` bit-identical, and the first MLP that can read block `i`'s attention is block
`i + 1`'s. Circuit and path-patching graphs on Pythia have no attention-to-MLP edge inside a
block.

```python
with model.trace(prompt):
    x = model.layers[2].input.save()
    attn = model.layers[2].self_attn.attention_output.save()
    mlp_in = model.layers[2].mlp.input.save()
    mlp = model.layers[2].mlp.mlp_output.save()
    out = model.layers[2].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)

with model.trace(prompt):
    model.layers[2].self_attn.attention_output[:] = 0
    mlp_in_ablated = model.layers[2].mlp.input.save()

assert torch.equal(mlp_in, mlp_in_ablated)
```

The block returns a bare tensor, so `model.layers[i].output` is `layer_output`.
`post_attention_layernorm` still runs after the attention returns, so in one trace read the
attention's values before `mlp.input`.

## Queries, keys and values come from one projection, laid out per head

`self_attn.query_key_value` is a single `Linear` from `hidden_size` to `3 * hidden_size`, with a
bias. Its output is grouped by head, `[q_h | k_h | v_h]` for each head `h` in turn, not `[Q | K | V]`
in three blocks: the first `hidden_size` columns are not the queries. Split it by viewing it as
`[batch, seq, num_heads, 3 * head_dim]`. Keys and values have `num_heads` heads; there is no
grouping.

`attention_queries` and `attention_keys` are read after the rotary embedding,
`attention_values` as split. For the queries and keys before the rotary, read the projection's
output in the same trace:

```python
with model.trace(prompt):
    qkv = model.layers[2].self_attn.source.self_query_key_value_0.output.save()
    q = model.layers[2].self_attn.attention_queries.save()

b, s, _ = qkv.shape
per_head = qkv.view(b, s, model.num_heads, 3 * model.head_dim).transpose(1, 2)
q_pre, k_pre, v = per_head.chunk(3, dim=-1)
rot = model.head_dim // 4                            # partial_rotary_factor 0.25
assert torch.equal(q_pre[..., rot:], q[..., rot:])   # past rot: no rotation
```

## Rotary turns a quarter of each head

The Pythia configs set `rotary_pct: 0.25`, which transformers reads as
`rope_parameters["partial_rotary_factor"]`. On 70m-deduped that is the first 16 of each head's
64 dimensions, rotated as two halves of 8 (`rotate_half`); the other 48 dimensions of every query
and key are the projection's output unchanged. Only those 16 dimensions make a query-key score
depend on the distance between the two tokens; the other 48 add the same term at any distance.
The scale is
`head_dim ** -0.5`, and `attention_scores` are the scaled, masked scores at the softmax's input, so
`softmax(attention_scores)` equals `attention_probabilities`. Other `GPTNeoXForCausalLM`
checkpoints set other fractions (`RedPajama-INCITE-Base-3B-v1` rotates whole heads, `1.0`).

## Load with eager for the interior

Under the default `sdpa` load the six interior values report unavailable, and under
`attn_implementation="eager"` all are served. The two implementations compute the same attention
(no softcap, sink or window here), so the default load is the same model up to rounding. The
Pythia files are `float16`, and a load without `dtype` keeps that, on CPU too; pass
`dtype=torch.float32` for exact comparisons.

```python
model = StandardizedTransformer(
    "EleutherAI/pythia-70m-deduped", dispatch=True, attn_implementation="eager"
)
```

## The final LayerNorm's bias is a fixed logit vector

`logits` equals `lm_head.output`, and `project_on_vocab` applied to the last block's
`layer_output` equals `logits` exactly. The final norm is a LayerNorm with a bias, so a logit lens
adds `lm_head.weight @ norm.bias` at every block, whatever the input. On 70m-deduped that vector is
large, a standard deviation of 41 logits against 47 for a whole lens readout at block 3 on a
sample prompt, and its top tokens are `,`, `.`, `\\n`, ` that`, ` to`. Folding the norm into the
unembedding has to keep it as a bias term.

```python
bias_logits = model.lm_head.weight @ model.norm.bias
top = [model.tokenizer.decode(i) for i in bias_logits.topk(5).indices.tolist()]
```

`embed_in` and `lm_head` are separate matrices, so the unembedding is not the embedding's
transpose and an edit to one leaves the other. The vocabulary is padded: 50304 rows on
70m-deduped against 50277 tokenizer tokens. The 27 extra ids decode to `''`; their unembedding
rows are not zero, so they get logits.

## The tokenizer adds no BOS

A prompt is tokenized without a leading token: position 0 is the prompt's first token, not a
fixed BOS. `<|endoftext|>` (id 0) is both `bos_token` and `eos_token`, and `<|padding|>` (id 1)
is the pad token.

## The Pythia suite: sizes, deduplication, training steps

Pythia is ten sizes, 14m to 12b, each trained on the Pile and, as `-deduped`, on the Pile after
global deduplication; the model cards state that every model saw the same data in the same order.
Every config sets the parallel block, `rotary_pct: 0.25` and untied embeddings. Each repo has 154
training checkpoints as branches, `step0`, `step1`, `step2`, `step4` ... `step512`, then every
1000 steps to `step143000`; load one with `revision`. EleutherAI also publishes retrained seeds
(`pythia-70m-seed1`, ...) and the `-v0` runs.

```python
model = StandardizedTransformer(
    "EleutherAI/pythia-70m-deduped", revision="step1000", dispatch=True
)
```

Other checkpoints load as `GPTNeoXForCausalLM` with different settings:
`EleutherAI/gpt-neox-20b` (parallel, `gelu_fast`), `stabilityai/stablelm-base-alpha-3b`
(parallel), `togethercomputer/RedPajama-INCITE-Base-3B-v1` (sequential, full rotary).

## Sparse autoencoders and tuned lenses

- `saprmarks/pythia-70m-deduped-saes`, the dictionaries of *Sparse Feature Circuits*: `embed`,
  `attn_out_layerN`, `mlp_out_layerN` and `resid_out_layerN` read the outputs of `embed_in`, the
  attention, the MLP and the block, which are `model.token_embeddings`,
  `model.layers[N].self_attn.attention_output`, `model.layers[N].mlp.mlp_output` and
  `model.layers[N].layer_output`.
- `EleutherAI/sae-pythia-70m-deduped-32k` and `EleutherAI/sae-pythia-160m-deduped-32k`: the
  folders `layers.N.attention`, `layers.N.mlp` and `layers.N` are SAEs on those modules' outputs,
  `attention_output`, `mlp_output` and `layer_output`.
- Tuned lenses for the deduped Pythia models and GPT-NeoX-20B, in the `AlignmentResearch/tuned-lens`
  Hugging Face space that the `tuned-lens` package loads from. A lens for block `N` reads
  transformers' `hidden_states[N]`, which is `model.layers[N].input`, and decodes through the
  model's own final norm and unembedding, as `project_on_vocab` does.

On 70m-deduped at block 3, both sets of SAEs reconstruct the value named above (fraction of
variance unexplained 0.02 to 0.39) and not the sublayer's input (`self_attn.input`, `mlp.input`:
above 1).
"""
