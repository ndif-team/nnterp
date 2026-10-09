"""GPT-2: LayerNorm, learned absolute positions, a fused query/key/value projection."""

MODEL_TYPE = "gpt2"
TITLE = "GPT-2"
SUBTITLE = (
    "Llama's sequential block with LayerNorm in place of RMSNorm, a learned position embedding added "
    "to the token embeddings before block 0, and queries, keys and values cut from one fused c_attn projection."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "openai-community/gpt2"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-internal-testing/tiny-random-gpt2"
CHECKPOINTS = [
    "openai-community/gpt2", "openai-community/gpt2-medium",
    "openai-community/gpt2-large", "openai-community/gpt2-xl",
    "distilbert/distilgpt2",
    "stanford-crfm/alias-gpt2-small-x21", "stanford-crfm/battlestar-gpt2-small-x49",
    "stanford-crfm/caprica-gpt2-small-x81", "stanford-crfm/darkmatter-gpt2-small-x343",
    "stanford-crfm/expanse-gpt2-small-x777",
    "stanford-crfm/arwen-gpt2-medium-x21", "stanford-crfm/beren-gpt2-medium-x49",
    "stanford-crfm/celebrimbor-gpt2-medium-x81", "stanford-crfm/durin-gpt2-medium-x343",
    "stanford-crfm/eowyn-gpt2-medium-x777",
]

#: Set by hues.py (lineage: GPT-2).
PALETTE = {"hue": 35}
VLLM = True
QUIRKS = ["position-embeddings", "layernorm", "fused-qkv"]

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
            "detail": "{num_heads} heads × {head_dim}, fused c_attn",
            "pre_norm_note": "Native name ln_1: a LayerNorm with a bias, which subtracts the mean before scaling.",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {activation_function} (c_fc, c_proj)",
            "pre_norm_note": "Native name ln_2, the MLP's input norm as on Llama: a LayerNorm with a bias.",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "token_embeddings is wte's lookup alone. The learned position embedding transformer.wpe is added "
             "after it, so layers[0].input = token_embeddings + transformer.wpe.output.",
    "norm": "ln_f is a LayerNorm with a bias: it subtracts the mean, and its bias adds the same logit offset "
            "(lm_head applied to the bias) at every position.",
    "head": "lm_head has no bias and shares its weight with embed_tokens (wte). Its output is logits, "
            "unchanged; project_on_vocab applies ln_f, then lm_head.",
}

NOTES = """
## The block, in order

```
h   = x + attn(ln_1(x))      # ln_1 = input_layernorm
out = h + mlp(ln_2(h))       # ln_2 = post_attention_layernorm
```

The block returns a bare tensor. Nothing sits between a sublayer and the stream:
`attention_output` is `self_attn.output[0]` (after `c_proj`) and `mlp_output` is `mlp.output`
(after the MLP's `c_proj`), and the identity
`layers[i].input + attention_output + mlp_output == layer_output` is exact in float32 (difference
0.0 on `gpt2`). Both norms are `LayerNorm` with a weight and a bias; `ln_2` is the MLP's input norm,
as Llama's `post_attention_layernorm` is.

## Position embeddings are added after embed_tokens

`token_embeddings` is `wte`'s lookup only. The learned position embedding `transformer.wpe`, one row
per position up to `n_positions` (1024), is added to it before block 0, so the stream entering block 0
is the sum, and an edit to `token_embeddings` leaves the position term in place:

```python
with model.trace(prompt):
    tokens = model.token_embeddings.save()
    positions = model.transformer.wpe.output.save()
    x0 = model.layers[0].input.save()

torch.equal(tokens + positions, x0)        # True
```

A prompt longer than `n_positions` fails in `wpe` with `IndexError`. Positions are absolute, so a
left-padded row would read shifted rows of `wpe`; nnsight derives `position_ids` from the attention
mask for a left-padded batch, and a padded prompt gives the same logits as the prompt alone (largest
difference 1.3e-4 on `gpt2`) and the same greedy generation.

## The first position is a sink with a huge norm

On `gpt2` the stream at position 0 has a norm of 2,500 to 3,200 from block 2's output to block
10's, whatever the token, against medians of about 60 to 250 at the other positions. It is also where
attention goes: averaged over heads and later queries, 41% to 52% of the pattern lands on key 0 in
blocks 1 and 2, and 57% to 91% in blocks 3 to 11. Leave position 0 out of norms you scale a steering
vector by, of activation statistics and of SAE metrics.

The tokenizer adds no BOS token, so without one the first word of the prompt becomes the sink.
Writing `<|endoftext|>` (id 50256, the BOS and EOS token) at the start of the prompt gives the sink a
fixed token; it tokenizes to that one id. TransformerLens prepends it by default, so activations
recorded with its defaults have it at position 0. Words inside a sentence are tokens with
their leading space (`" Paris"` is 6342, `"Paris"` is 40313), and a prompt ending in a space ends
with the lone-space token 220.

## Queries, keys and values are cut from one Conv1D

The projections are `Conv1D`, whose weight is stored `[in, out]` (`c_attn.weight` is
`[768, 2304]` on `gpt2`): the module computes `x @ weight + bias`, the transpose of `nn.Linear`'s
layout. The queries', keys' and values' weights are column blocks of `c_attn.weight`, and head `h`
is columns `h * head_dim` to `(h + 1) * head_dim` within a block:

```python
attn = model.layers[3].self_attn
W_Q, W_K, W_V = attn.c_attn.weight.split(model.hidden_size, dim=1)
b_Q, b_K, b_V = attn.c_attn.bias.split(model.hidden_size)

with model.trace(prompt):
    h = model.layers[3].self_attn.input.save()                  # ln_1's output
    q = model.layers[3].self_attn.attention_queries.save()

heads = (model.num_heads, model.head_dim)
q_again = (h @ W_Q + b_Q).unflatten(-1, heads).transpose(1, 2)
torch.testing.assert_close(q_again, q)
```

The three are views of the split `c_attn` output, so torch refuses an in-place edit of
`attention_queries`, `attention_keys` or `attention_values`. Edit a copy and assign it:

```python
with model.trace(prompt):
    q = model.layers[3].self_attn.attention_queries.clone()
    q[:, 0] = 0                                                 # head 0's queries
    model.layers[3].self_attn.attention_queries = q
```

Every head has its own keys and values (`num_kv_heads == num_heads`), and the scores are scaled by
`head_dim ** -0.5` (0.125).

## Load with eager; two config flags change the attention

The default load runs `sdpa`, which has no queries, pattern or head outputs to read;
`attn_implementation="eager"` runs the shared eager forward where nnterp reads them.
`attention_output` and `mlp_output` are there under either load.

The Stanford CRFM checkpoints (`stanford-crfm/*-gpt2-small-*` and `*-gpt2-medium-*`, five seeds
each, with intermediate training checkpoints as revision tags such as `checkpoint-99000`) set
`reorder_and_upcast_attn` and `scale_attn_by_inverse_layer_idx`. The second divides block `i`'s
query scale by `i + 1`. The first takes GPT-2's own upcast path under eager, where nnterp reads
nothing inside the attention. In float32 that path computes what the shared one does: on
`alias-gpt2-small-x21` the logits of the two paths are bit-identical on CPU and agree within 1e-5 on
GPU. So a float32 load with the flag off gives the interior back without changing the forward:

```python
model = StandardizedTransformer(
    "stanford-crfm/alias-gpt2-small-x21", dispatch=True, dtype=torch.float32,
    attn_implementation="eager", reorder_and_upcast_attn=False,
)
```

`alias-gpt2-small-x21` loads in float16 by default, and there the paths differ, since the upcast one
computes the scores in float32 (largest logit difference 0.023 on GPU).

## The readout: LayerNorm with a bias, tied weights

`logits` is `lm_head.output` with nothing applied, and `project_on_vocab` is `lm_head(ln_f(x))`, so a
logit lens at the last block equals `logits` exactly. `ln_f` subtracts the mean over the hidden axis,
so adding the same number to every coordinate of the stream changes no logit. Its bias adds
`lm_head(ln_f.bias)` to every position's logits whatever the input: on `gpt2` that vector is highest
on `,`, ` the`, ` and`, `.` and newline, with a standard deviation of 0.9 across the vocabulary.

Folding the final norm into the unembedding takes the gain into `W_U` and the bias into a vocabulary
bias. With the scale frozen at the run's value the readout is linear, which is what direct logit
attribution needs: each contribution's term is its centred vector through the same map, and the terms
of `layers[0].input` and of every `attention_output` and `mlp_output`, plus the bias's, sum to the
logits (largest difference 1.3e-4 on `gpt2`, on logits of size near 100).

```python
ln_f, W_U = model.norm._module, model.lm_head.weight
with model.trace(prompt):
    resid = model.layers[-1].layer_output[0, -1].save()
    logits = model.logits[0, -1].save()

scale = (resid.var(unbiased=False) + ln_f.eps).sqrt()          # frozen
readout = lambda v: (ln_f.weight * (v - v.mean()) / scale) @ W_U.T
logits_again = readout(resid) + ln_f.bias @ W_U.T
torch.testing.assert_close(logits_again, logits, atol=1e-3, rtol=1e-5)
```

`lm_head` has no bias and its weight is `embed_tokens.weight`, the same tensor, so an edit to the
embedding matrix is an edit to the unembedding.

## The family's checkpoints

The four OpenAI sizes are 12 × 768 (`gpt2`), 24 × 1024 (`gpt2-medium`), 36 × 1280 (`gpt2-large`)
and 48 × 1600 (`gpt2-xl`, 25 heads), blocks × width; every size has `head_dim` 64, `gelu_new`, an
MLP four times the width, 1024 positions and the same 50257-token vocabulary. `distilgpt2` is 6
blocks of `gpt2`'s width. Every checkpoint whose config says `model_type: gpt2` takes this family
(DialoGPT's and ProtGPT2's do); check its config for the two flags above.

## TransformerLens hook names

Measured on `gpt2` against TransformerLens 2.18's `HookedTransformer.from_pretrained_no_processing`,
block 6, each pair equal to within 1e-4:

- `hook_embed`: `model.token_embeddings`
- `hook_pos_embed`: `model.transformer.wpe.output`
- `blocks.i.hook_resid_pre`: `model.layers[i].input`
- `blocks.i.attn.hook_q`, `hook_k`, `hook_v`: `self_attn.attention_queries`, `attention_keys`,
  `attention_values`, transposed (TransformerLens puts the position axis before the head axis)
- `blocks.i.attn.hook_attn_scores`: `self_attn.attention_scores`, on the unmasked entries
- `blocks.i.attn.hook_pattern`: `self_attn.attention_probabilities`
- `blocks.i.attn.hook_z`: `self_attn.attention_head_outputs`
- `blocks.i.hook_attn_out`: `self_attn.attention_output`
- `blocks.i.hook_resid_mid`: `model.layers[i].post_attention_layernorm.input`
- `blocks.i.mlp.hook_post`: `model.layers[i].mlp.act.output`
- `blocks.i.hook_mlp_out`: `mlp.mlp_output`
- `blocks.i.hook_resid_post`: `model.layers[i].layer_output`
- `ln_final.hook_normalized`: `model.norm.output`

`HookedTransformer.from_pretrained`, the default, rewrites the weights, and the hooks move with them.
It centres every weight that writes to the stream, so each residual hook and `hook_attn_out` /
`hook_mlp_out` equals the nnterp value minus its mean over the hidden axis. It folds `ln_1`'s bias
and the value bias out of the values, so `hook_v` and `hook_z` equal the nnterp values minus
`ln_1.bias @ W_V + b_V`, per head. `ln_final.hook_normalized` drops `ln_f`'s gain and bias, and the
logits are nnterp's minus their mean over the vocabulary. The queries, keys, scores, pattern and
`mlp.hook_post` are unchanged.

## Sparse autoencoders and neuron explanations

**`gpt2-small-res-jb`** (SAELens; Hub `jbloom/GPT2-Small-SAEs-Reformatted`) has 24576 latents on
`blocks.i.hook_resid_pre` for every block, plus `blocks.11.hook_resid_post`. They were trained on
TransformerLens' centred weights (SAELens loads them with `center_writing_weights`), so their input is
`model.layers[i].input` minus its mean over the hidden axis. On a 59-token text (position 0 left out),
block 8's SAE leaves 15% of the variance unexplained on the centred stream and 250% on the raw one.

```python
with model.trace(prompt):
    resid = model.layers[8].input.save()

sae_input = resid - resid.mean(-1, keepdim=True)      # gpt2-small-res-jb's input
```

**OpenAI's v5 SAEs** (`openai/sparse_autoencoder`; also in SAELens as `gpt2-small-resid-post-v5-32k`
and its kin) are TopK autoencoders with 32k and 128k latents at four places per block:
`resid_delta_attn` (`attention_output`), `resid_post_attn` (`post_attention_layernorm.input`),
`resid_delta_mlp` (`mlp_output`) and `resid_post_mlp` (`layer_output`). The repository's README
extracts activations with `center_writing_weights=False`, and the autoencoder layer-normalizes its
input itself, so the nnterp value goes in as it is: block 6's `resid_post_mlp` SAE reconstructs
`layers[6].layer_output` with a normalized MSE of 0.058 on the same text.

**OpenAI's neuron explanations** (`openai/automated-interpretability`) cover every MLP neuron of
`gpt2-xl` (48 blocks × 6400) and, in a second set, of `gpt2`. A neuron's activation is
`model.layers[i].mlp.act.output[..., n]`, after the `gelu_new`. The README says the published
activations were computed with a different GELU and differ from the exact ones by a median of 0.009
on `gpt2`.
"""
