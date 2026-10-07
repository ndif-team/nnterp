"""GPT-BigCode: BigCode's StarCoder and SantaCoder, and the fine-tunes that load as GPTBigCodeForCausalLM."""

MODEL_TYPE = "gpt_bigcode"
TITLE = "StarCoder / SantaCoder"
SUBTITLE = (
    "GPT-2's sequential block with multi-query attention: one key head and one value head serve every "
    "query head, and the queries, keys and values are cut from one fused c_attn projection."
)

#: The public checkpoint the sizes, config and support() on the page are read from (meta build, config only).
REFERENCE = "bigcode/gpt_bigcode-santacoder"
#: The tiny checkpoint the test suite builds the page from.
PINNED = "hf-tiny-v2/tiny-random-GPTBigCodeForCausalLM"
CHECKPOINTS = [
    "bigcode/gpt_bigcode-santacoder", "bigcode/tiny_starcoder_py",
    "bigcode/starcoderbase-1b", "bigcode/starcoderbase-3b", "bigcode/starcoderbase-7b",
    "bigcode/starcoderbase", "bigcode/starcoder", "bigcode/starcoderplus",
    "bigcode/octocoder", "HuggingFaceH4/starchat-alpha", "WizardLMTeam/WizardCoder-15B-V1.0",
]

#: Set by hues.py (lineage: GPT-2).
PALETTE = {"hue": 38}
VLLM = False
QUIRKS = ["position-embeddings", "layernorm", "fused-qkv"]

#: What the visualization draws: GPT-2's sequential block, ln_1 before the attention, ln_2 before the MLP.
BLOCK = {
    "topology": "sequential",
    "sublayers": [
        {
            "host": "self_attn",
            "kind": "attention",
            "label": "Attention",
            "pre_norm": "input_layernorm",
            "pre_norm_note": "Native name ln_1: a LayerNorm with a bias, which subtracts the mean before scaling.",
            "contribution": "attention_output",
            "interior": [
                "attention_queries", "attention_keys", "attention_values",
                "attention_scores", "attention_probabilities", "attention_head_outputs",
            ],
            "detail": "{num_heads} q heads, {num_kv_heads} kv × {head_dim}",
        },
        {
            "host": "mlp",
            "kind": "mlp",
            "label": "MLP",
            "pre_norm": "post_attention_layernorm",
            "pre_norm_note": "Native name ln_2, the MLP's input norm: a LayerNorm with a bias.",
            "contribution": "mlp_output",
            "detail": "{hidden_size} → {intermediate_size} → {hidden_size}, {activation_function}",
        },
    ],
}

#: Notes on the model-level strip, by node.
STRIP = {
    "embed": "token_embeddings is wte's lookup alone. The learned position embedding transformer.wpe is added "
             "after it, so layers[0].input = token_embeddings + transformer.wpe.output.",
    "norm": "ln_f is a LayerNorm with a bias; project_on_vocab applies it, bias included.",
    "head": "lm_head has no bias and shares its weight with embed_tokens (wte). Its output is logits, unchanged.",
}

NOTES = """
## The block, in order

```
h   = x + attn(ln_1(x))      # ln_1 = input_layernorm
out = h + mlp(ln_2(h))       # ln_2 = post_attention_layernorm
```

The block returns a bare tensor. `attention_output` is `self_attn.output[0]` (after `c_proj`) and
`mlp_output` is `mlp.output` (after the MLP's `c_proj`); nothing sits between either and the stream,
and `layers[i].input + attention_output + mlp_output == layer_output` is exact (bit-identical on the
pinned checkpoint). The attention runs transformers' shared eager forward, where nnterp reads the
six interior values.

## One key head and one value head serve every query head

Every released checkpoint sets `multi_query`: the attention projects `num_heads` query heads and a
single key head and value head, which every query head reads. `model.num_kv_heads` is 1,
`attention_keys` and `attention_values` are `[batch, 1, seq, head_dim]`, and an edit to them reaches
every head's scores or outputs at once. To change what one head reads, edit its queries, its scores
or its pattern instead. Head `h`'s scores are `q[:, h] @ k[:, 0]ᵀ * head_dim ** -0.5` (0.0884 at
`head_dim` 128); the softmax runs in float32 and is cast back to the queries' dtype.

```python
with model.trace(prompt):
    q = model.layers[1].self_attn.attention_queries.save()   # num_heads heads
    k = model.layers[1].self_attn.attention_keys.save()      # one head
    scores = model.layers[1].self_attn.attention_scores.save()

n = scores.shape[-1]
causal = torch.ones(n, n, dtype=torch.bool, device=scores.device).tril()
again = q @ k.transpose(-1, -2) * model.head_dim ** -0.5   # k broadcasts
torch.testing.assert_close(again[..., causal], scores[..., causal])
```

The configs' `attention_softmax_in_fp32` and `scale_attention_softmax_in_fp32` are not read by
transformers' forward: the eager softmax is float32 whatever they say.

## Queries, keys and values are cut from one Linear

`c_attn` is an `nn.Linear` from `hidden_size` to `hidden_size + 2 * head_dim`, its weight stored
`[out, in]`. Its output is laid out as all the query heads, then the one key head, then the one
value head, and within the queries head `h` is columns `h * head_dim` to `(h + 1) * head_dim`:

```python
E, d = model.hidden_size, model.head_dim
W_Q, W_K, W_V = model.layers[1].self_attn.c_attn.weight.split([E, d, d])
b_Q, b_K, b_V = model.layers[1].self_attn.c_attn.bias.split([E, d, d])

with model.trace(prompt):
    h = model.layers[1].self_attn.input.save()               # ln_1's output
    q = model.layers[1].self_attn.attention_queries.save()

q_again = (h @ W_Q.T + b_Q).unflatten(-1, (model.num_heads, d)).transpose(1, 2)
torch.testing.assert_close(q_again, q)
```

With `multi_query` false, transformers lays `c_attn`'s output out per head instead, as
`[q | k | v]` for each of `num_heads` heads.

## Assign the queries; do not edit them in place

The queries are a view of the split `c_attn` output, so with gradients on torch refuses an in-place
edit of `attention_queries` (`RuntimeError: ... is a view and is being modified inplace`). Edit a
copy and assign it, or edit under `torch.no_grad()`. The keys and values take an in-place edit.

```python
with model.trace(prompt):
    q = model.layers[1].self_attn.attention_queries.clone()
    q[:, 0] = 0                                              # head 0's queries
    model.layers[1].self_attn.attention_queries = q
```

## Load with eager

The default load runs `sdpa`, and the six interior values are unavailable under it;
`attn_implementation="eager"` serves them. `attention_output` and `mlp_output` are there under
either load.

## Position embeddings are added after embed_tokens

`token_embeddings` is `wte`'s lookup only. The learned position embedding `transformer.wpe`, one
row per position up to `n_positions` (2048 on SantaCoder, 8192 on StarCoder and
`tiny_starcoder_py`), is added to it before block 0, so an edit to `token_embeddings` leaves the
position term in place:

```python
with model.trace(prompt):
    tokens = model.token_embeddings.save()
    positions = model.transformer.wpe.output.save()
    x0 = model.layers[0].input.save()

assert torch.equal(tokens + positions, x0)
```

The tokenizer adds no BOS, so position 0 is the prompt's first token. On `tiny_starcoder_py`, on a
58-token Python function, blocks 12 to 19 put 54% to 80% of their pattern (averaged over heads and
the later queries) on key 0.

## The readout: LayerNorm with a bias, tied weights

`logits` is `lm_head.output` with nothing applied, and `project_on_vocab` applied to the last
block's `layer_output` equals `logits` exactly (difference 0.0 on `tiny_starcoder_py`). `lm_head`
has no bias and its weight is `embed_tokens.weight`, the same tensor. `ln_f`'s bias adds
`lm_head.weight @ norm.bias` to every position's logits whatever the input (a standard deviation of
0.12 logits on `tiny_starcoder_py`).

## The family's checkpoints

SantaCoder (`gpt_bigcode-santacoder`, 1.1B) is 24 blocks × 2048 with 16 query heads of 128 and
`gelu_pytorch_tanh`. `tiny_starcoder_py` is 20 × 768 with 12 query heads of 64. StarCoder,
StarCoderBase (with its 1B, 3B and 7B sizes) and StarCoderPlus are gated, so the page greys them
without access; their fine-tunes OctoCoder, StarChat-alpha and WizardCoder-15B are open, and are
40 × 6144 with 48 query heads of 128 and `gelu`. Every one has an MLP four times the width
(`n_inner`) and one key/value head. `bigcode/santacoder` is SantaCoder's original repository, a
`gpt2` config with remote code (`GPT2LMHeadCustomModel`), so it does not load as this family;
`gpt_bigcode-santacoder` does.
"""
